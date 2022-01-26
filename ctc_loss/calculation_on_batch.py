import torch


def find_alpha_batched(outs, labels, target_lens):
    labels = labels.to(torch.device(outs.device))
    loss = torch.tensor(0., requires_grad=False, device=torch.device(outs.device))
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1

    alpha = torch.zeros((batch_size, T, labels.shape[1]), device=torch.device(outs.device))
    alpha[:, 0, 0] = torch.stack([outs[x, 0, labels[x, 0]] for x in range(batch_size)])
    alpha[:, 0, 1] = torch.stack([outs[x, 0, labels[x, 1]] for x in range(batch_size)])

    c = alpha[:, 0, 0] + alpha[:, 0, 1]
    if torch.equal(torch.count_nonzero(c), torch.tensor(0, device=torch.device(outs.device))):
        print('c after init is zero!')
    c[c == 0.] = 1
    alpha[:, 0] /= (c.reshape(-1, 1))
    loss += torch.log(c).mean()

    # common mask for reds and blues as labels have blanks
    red_blue_mask = torch.tensor(labels[:, :-2] == labels[:, 2:], device=torch.device(outs.device))

    for t in range(1, T):
        ss = torch.maximum(torch.tensor(0., device=torch.device(outs.device)), L - 2 * (T - t))
        # ss = torch.maximum(torch.tensor(0., device=torch.device(outs.device)), L - 2 * (T - t) + 1)
        # e = torch.minimum(torch.tensor(2 * t + 1, device=torch.device(outs.device)), L)
        e = torch.minimum(torch.tensor(2 * t + 2, device=torch.device(outs.device)), L)

        alpha[:, t, 0] = alpha[:, t - 1, 0]
        alpha[:, t, 1:] = alpha[:, t - 1, 1:] + alpha[:, t - 1, :-1]
        orange = torch.clone(alpha[:, t - 1, :-2])
        orange[red_blue_mask] = 0
        alpha[:, t, 2:] += orange
        alpha[:, t] *= torch.gather(outs[:, t, :], 1, labels)

        j_indices = torch.cat([torch.arange(ss[i].int(), e[i].int(), device=torch.device(outs.device)) \
                               for i in range(batch_size)]).type(torch.LongTensor)
        i_indices = torch.cat([torch.tensor((i,) * (e[i].int() - ss[i].int()), device=torch.device(outs.device)) \
                               for i in range(batch_size)]).type(torch.LongTensor)

        mask = torch.full((batch_size, labels.shape[1]), True, device=torch.device(outs.device))
        mask[i_indices, j_indices] = False
        alpha[:, t, :][mask] *= 0
        c = alpha[:, t, :].sum(dim=-1)
        c[c == 0.] = 1
        alpha[:, t] /= (c.reshape(-1, 1))
        loss += torch.log(c).mean()

    if torch.equal(torch.count_nonzero(alpha), torch.tensor(0, device=torch.device(outs.device))):
        print('alpha is zero!')

    return alpha, -loss


# workable
def find_beta_batched(outs, labels, target_lens):
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1

    betta = torch.zeros((batch_size, T, labels.shape[1]), device=torch.device(outs.device))

    for sample in range(batch_size):
        betta[sample, -1, L[sample] - 1] = outs[sample, -1, labels[sample, L[sample] - 1]]
        betta[sample, -1, L[sample] - 2] = outs[sample, -1, labels[sample, L[sample] - 2]]
        d = betta[sample, -1, L[sample] - 1] + betta[sample, -1, L[sample] - 2]
        d = d if torch.is_nonzero(d) else torch.tensor(1., device=torch.device(outs.device))
        betta[sample, -1, :] /= d

    if torch.equal(torch.count_nonzero(betta), torch.tensor(0, device=torch.device(outs.device))):
        print('betta after init is zero!')

    red_blue_mask = torch.tensor(labels[:, :-2] == labels[:, 2:], device=torch.device(outs.device))

    for t in range(T - 1)[::-1]:
        labels = labels.to(torch.device(outs.device))
        ss = torch.maximum(torch.tensor(0., device=torch.device(outs.device)), L - 2 * (T - t))
        # ss = torch.maximum(torch.tensor(0., device=torch.device(outs.device)), L - 2 * (T - t) + 1)
        e = torch.minimum(torch.tensor(2 * t + 2, device=torch.device(outs.device)), L)
        # e = torch.minimum(torch.tensor(2 * t + 1, device=torch.device(outs.device)), L)

        betta[:, t, -1] = betta[:, t + 1, -1]
        betta[:, t, :-1] = betta[:, t + 1, :-1] + betta[:, t + 1, 1:]
        orange = torch.clone(betta[:, t + 1, 2:])

        orange[red_blue_mask] = 0
        betta[:, t, :-2] += orange
        betta[:, t] *= torch.gather(outs[:, t, :], 1, labels)

        j_indices = torch.cat([torch.arange(ss[i].int(), e[i].int(), device=torch.device(outs.device)) \
                               for i in range(batch_size)]).type(torch.LongTensor)
        i_indices = torch.cat([torch.tensor((i,) * (e[i].int() - ss[i].int()), device=torch.device(outs.device)) \
                               for i in range(batch_size)]).type(torch.LongTensor)

        mask = torch.full((batch_size, labels.shape[1]), True)
        mask[i_indices, j_indices] = False

        betta[:, t, :][mask] *= 0
        d = betta[:, t, :].sum(dim=-1)
        d[d == 0.] = 1
        betta[:, t] /= (d.reshape(-1, 1))

    if torch.equal(torch.count_nonzero(betta), torch.tensor(0, device=torch.device(outs.device))):
        print('betta is zero! t:')

    return betta


# workable
def compute_grad_batched(outs, labels, a, b, target_lens):
    L = 2 * target_lens + 1
    T = outs.shape[1]
    batch_size = outs.shape[0]
    k = outs.shape[-1]

    ab = a * b

    if torch.equal(torch.count_nonzero(outs), torch.tensor(0, device=torch.device(outs.device))):
        print('outs in grad is zero!!')
    if torch.equal(torch.count_nonzero(ab), torch.tensor(0, device=torch.device(outs.device))):
        print('ab is zero!!')
    grads = torch.zeros((batch_size, T, k), device=torch.device(outs.device))

    for sample in range(batch_size):
        for s in range(L[sample]):
            grads[sample, :, labels[sample, s]] += ab[sample, :, s]
            ab[sample, :, s] /= (outs[sample, :, labels[sample, s]])

    absum = torch.sum(ab, dim=-1)
    grads = outs - grads / (outs * absum.unsqueeze(-1))
    grads[torch.isnan(grads)] = 0.

    return grads
