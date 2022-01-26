import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import os
from time import time
from matplotlib import pyplot as plt

from data.dataloader import Data_loader
from training.config import cfg
from model.text_recognition_model import Model
from ctc_loss.ctc_loss import myCTCloss
from metrics.accuracy import calc_acc, decoded_labels

torch.manual_seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Train:
    def __init__(self, trainloader, valloader, testloader, config, own_loss=False, ctc_loss_batched=False):
        self.config = config
        self.train_iterator = trainloader
        self.val_iterator = valloader
        self.test_iterator = testloader

        self.model = Model(use_log_softmax=not own_loss)
        self.model.to(config.train.device)
        self.own_loss = own_loss
        self.crit = myCTCloss(batched=ctc_loss_batched) if own_loss else nn.CTCLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=config.train.init_lr, momentum=0.9,
                               nesterov=True)
        self.scheduler = ReduceLROnPlateau(self.optim, patience=config.train.patience)

        self.nrof_epochs = config.train.nrof_epochs
        self.epoch_size = config.train.train_size // config.train.batch_size + 1
        self.cur_epoch, self.global_step = 0, 0

        self.best_acc = 0
        self.best_loss = 1000.
        self.tr_losses = []
        self.tr_accs = []

    def save_model(self):
        if not os.path.exists(os.path.dirname(self.config.train.ckpt_path)):
            os.makedirs(os.path.dirname(self.config.train.ckpt_path))

        torch.save({"step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "acc": self.best_acc,
                    "ctc_loss": self.best_loss},
                   self.config.train.ckpt_path)

        print("Model saved...")

    def load_model(self):
        # ckpt = torch.jit.load(self.conf.train.ckpt_path, map_location=cfg.train.device)
        ckpt = torch.load(self.config.train.ckpt_path, map_location=cfg.train.device)
        self.cur_epoch = ckpt["step"] // self.epoch_size
        self.global_step = ckpt["step"] + 1
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])
        for g in self.optim.param_groups:
            g['lr'] = cfg.train.init_lr
        # self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_acc = ckpt["acc"]
        self.best_loss = ckpt["ctc_loss"]
        print("Model loaded...")

    def train_epoch(self, to_save=False):
        self.model.train()
        nrof_correct, nrof_samples, cur_loss = 0, 0, 0.0

        for batch_idx, batch in enumerate(self.train_iterator):
            inputs, labels = batch[0].to(self.config.train.device), batch[1].to(self.config.train.device)
            target_lens = batch[2].to(self.config.train.device)

            self.optim.zero_grad()
            outputs = self.model(inputs)
            if to_save:
                torch.save(labels, 'labels_1')
                torch.save(outputs, 'outputs_1')

            input_lengths = torch.full(size=(self.config.train.batch_size,),
                                       fill_value=outputs.shape[3], dtype=torch.long)

            if self.own_loss:
                sftmx_out = nn.functional.softmax(outputs)
                loss, grads = self.crit(sftmx_out.squeeze().permute(0, 2, 1), labels, target_lens)

                if torch.isnan(loss):
                    print('grads\n', grads)

                outputs.backward(gradient=grads)
            else:
                loss = self.crit(outputs.squeeze().permute(2, 0, 1), labels, input_lengths, target_lens)
                loss.backward()

            cur_loss += loss.item()
            nrof_correct += calc_acc(outputs.detach().cpu().numpy().squeeze().argmax(axis=1), labels.cpu().numpy())
            nrof_samples += len(labels)

            self.optim.step()
            self.global_step += 1

            if batch_idx % self.config.train.log_interval == 0:
                tr_loss = cur_loss / nrof_samples
                tr_acc = float(nrof_correct) / nrof_samples
                print("Train ctc_loss: {:.4f}\nTrain acc: {:.2f}".format(tr_loss, tr_acc))

                self.tr_losses.append(tr_loss)
                self.tr_accs.append(tr_acc)

                nrof_correct, nrof_samples, cur_loss = 0, 0, 0.0
                print('Decoded true-predicted labels')
                print(dict(zip(decoded_labels(labels.cpu().numpy()),
                               decoded_labels(outputs.detach().cpu().numpy().squeeze().argmax(axis=1)))))
                print()
                print('lr', self.optim.state_dict()["param_groups"][0]["lr"])

            break

    def validate(self, phase='val', to_load=False):
        if to_load:
            self.load_model()
        self.model.eval()
        iterator = self.val_iterator if phase == 'val' else self.test_iterator
        nrof_correct, nrof_samples, total_nrof_correct = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                inputs, labels = batch[0].to(self.config.train.device), batch[1].to(self.config.train.device)
                outputs = self.model(inputs)
                total_nrof_correct += calc_acc(outputs.detach().cpu().numpy().squeeze().argmax(axis=1),
                                               labels.cpu().numpy())
                nrof_samples += len(labels)

                if batch_idx % self.config.train.log_interval == 0 and not batch_idx == 0:
                    print('Batch number {}/{}'.format(batch_idx, len(iterator)))
                    print(float(total_nrof_correct) / nrof_samples)
                    print('Decoded true-predicted labels')
                    print(dict(zip(decoded_labels(labels.cpu().numpy()),
                                   decoded_labels(outputs.detach().cpu().numpy().squeeze().argmax(axis=1)))))
                    print()

        total_acc = float(total_nrof_correct) / nrof_samples
        print('Total accuracy: ', total_acc)
        return total_acc

    def train(self):
        if self.config.train.load:
            self.load_model()

        for epoch in range(self.cur_epoch, self.cur_epoch + self.nrof_epochs):
            t = time()
            self.train_epoch()
            print("Epoch {} trained after {} sec".format(epoch, time() - t))

            # val_loss, val_acc = self.validate()
            # print('Validation ctc_loss: {}\nValidation accuracy:{}'.format(val_loss, val_acc))

        # self.save_model()

        return

    def show_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(self.tr_losses)
        ax1.set_title('Train losses')
        ax2.plot(self.tr_accs)
        ax2.set_title('Train accuracies')
        plt.show()


if __name__ == '__main__':
    t = time()
    DL = Data_loader(cfg)
    train_loader, val_loader, test_loader = DL.tr_loader, DL.val_loader, DL.test_loader
    Tr_model = Train(train_loader, val_loader, test_loader, cfg, own_loss=True, ctc_loss_batched=True)
    Tr_model.validate(phase='val', to_load=True)

    # Tr_model.train()
    # Tr_model.show_metrics()
    print('Time:', time() - t)
