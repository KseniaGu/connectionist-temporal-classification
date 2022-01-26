import torch
import torch.nn as nn

from utils import add_blanks, show_alpha_betta
from ctc_loss.calculation_on_batch import find_alpha_batched, find_beta_batched, compute_grad_batched
from ctc_loss.calculation_on_sample import find_alpha_with_blanks, find_beta_with_blanks, compute_grad_with_blanks


class myCTCloss(nn.Module):
    def __init__(self, batched=False):
        super(myCTCloss, self).__init__()
        self.batched = batched

    def forward(self, outputs, labels, target_lens, compute_grad=True):
        """
        Args:
            outputs (torch.tensor of shape (batch_size, T, alphabet_len)): softmax outputs
            labels (torch.tensor of shape (batch_size, L)): labels with indices without blanks, zero-padded to max size
        """

        smoothed_outputs = outputs.clone().detach()
        grads = None
        if torch.equal(torch.count_nonzero(smoothed_outputs), torch.tensor(0, device=torch.device(outputs.device))):
            print('outs all zero')
        #    print(smoothed_outputs)
        # smoothed_outputs[outputs == 0.] += eps
        # smoothed_outputs[outputs == 1.] -= eps

        if self.batched:
            # adding blanks through one, so that labels[sample_idx, 0] = 0
            blank_labels = torch.stack([add_blanks(label) for label in labels])
            self.alpha, loss = find_alpha_batched(smoothed_outputs, blank_labels, target_lens)
            self.betta = find_beta_batched(smoothed_outputs, blank_labels, target_lens)

            if compute_grad:
                grads = compute_grad_batched(smoothed_outputs, blank_labels, self.alpha, self.betta, target_lens)
                grads = grads.permute(0, 2, 1).unsqueeze(2)
        else:
            blank_labels = torch.stack([add_blanks(label) for label in labels])
            # self.alpha, ctc_loss = find_alpha(smoothed_outputs, labels, target_lens)
            self.alpha, loss = find_alpha_with_blanks(smoothed_outputs, blank_labels, target_lens)
            # self.betta = find_betta(smoothed_outputs, labels, target_lens)
            self.betta = find_beta_with_blanks(smoothed_outputs, blank_labels, target_lens)
            # grads = compute_grad(smoothed_outputs, labels, self.alpha, self.betta, target_lens)
            if compute_grad:
                grads = compute_grad_with_blanks(smoothed_outputs, blank_labels, self.alpha, self.betta, target_lens)
                grads = grads.permute(0, 2, 1).unsqueeze(2)

        return loss, grads

    def alpha_beta_examples(self, labels, outputs):
        image_num = 0
        labels = labels[image_num].unsqueeze(0)
        outputs = outputs[image_num].unsqueeze(0)

        print(f'Labels: {labels}\n\nOutputs:{outputs}')
        sft_max = nn.functional.softmax(outputs)

        Loss = myCTCloss(batched=True)
        loss, grads = Loss(sft_max.squeeze(2).permute(0, 2, 1), labels, torch.tensor([4]))
        show_alpha_betta(Loss.alpha, Loss.betta)
        print('Grads\n', grads)