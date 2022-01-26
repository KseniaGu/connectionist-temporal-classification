import torch

def find_alpha(outs, labels, target_lens):
    loss = torch.tensor(0., requires_grad=False, device=torch.device(outs.device))
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1
    nrof_samples = 0

    alpha = torch.zeros((batch_size, T, labels.shape[1]), requires_grad=False, device=torch.device(outs.device))
    for x in range(batch_size):
        alpha[x, 0, 0] = outs[x, 0, 0]
        alpha[x, 0, 1] = outs[x, 0, labels[x, 0]]

    c = alpha[:, 0, 0] + alpha[:, 0, 1]
    c[c == 0.] = 1

    alpha[:, 0] /= c.reshape(-1, 1)
    loss += torch.log(c).mean()

    for t in range(1, T):
        for sample in range(batch_size):
            ss = max(0, L[sample] - 2 * (T - t))
            e = min(2 * t + 2, L[sample])

            for s in range(ss, L[sample]):
                i = (s - 1) // 2
                red = alpha[sample, t - 1, s]
                blue = 0
                if s > 0:
                    blue = alpha[sample, t - 1, s - 1]

                if s % 2 == 0:
                    alpha[sample, t, s] = (red + blue) * outs[sample, t, 0]
                elif s == 1 or labels[sample, i] == labels[sample, i - 1]:
                    alpha[sample, t, s] = (red + blue) * outs[sample, t, labels[sample, i]]
                else:
                    orange = alpha[sample, t - 1, s - 2]
                    alpha[sample, t, s] = (red + blue + orange) * outs[sample, t, labels[sample, i]]

            c = torch.sum(alpha[sample, t, ss:e])

            if not torch.equal(c, torch.tensor(0.)):
                alpha[sample, t, ss:e] /= c
                loss += torch.log(c)
                nrof_samples += 1

    return alpha, - loss / nrof_samples


def find_alpha_with_blanks(outs, labels, target_lens):
    loss = torch.tensor(0., requires_grad=False, device=torch.device(outs.device))
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1

    alpha = torch.zeros((batch_size, T, labels.shape[1]), requires_grad=False, device=torch.device(outs.device))
    alpha[:, 0, 0] = torch.stack([outs[x, 0, labels[x, 0]] for x in range(batch_size)])
    alpha[:, 0, 1] = torch.stack([outs[x, 0, labels[x, 1]] for x in range(batch_size)])

    c = alpha[:, 0, 0] + alpha[:, 0, 1]
    c[c == 0.] = 1
    alpha[:, 0] /= c.reshape(-1, 1)
    loss += torch.log(c).mean()

    for t in range(1, T):
        for sample in range(batch_size):
            ss = max(0, L[sample] - 2 * (T - t))
            e = min(2 * t + 2, L[sample])

            for s in range(ss, L[sample]):
                red = alpha[sample, t - 1, s]
                blue = alpha[sample, t - 1, s - 1] if s > 0 else 0

                if s % 2 == 0 or s == 1 or labels[sample, s] == labels[sample, s - 2]:
                    alpha[sample, t, s] = (red + blue) * outs[sample, t, labels[sample, s]]
                else:
                    orange = alpha[sample, t - 1, s - 2]
                    alpha[sample, t, s] = (red + blue + orange) * outs[sample, t, labels[sample, s]]

            c = torch.sum(alpha[sample, t, ss:e])

            c = c if torch.is_nonzero(c) else torch.tensor(1.)
            alpha[sample, t, :] /= c
            loss += torch.log(c)

    return alpha, - loss / batch_size


def find_beta(outs, labels, target_lens):
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1

    betta = torch.zeros((batch_size, T, labels.shape[1]), requires_grad=False, device=torch.device(outs.device))

    for x in range(batch_size):
        betta[x, -1, L[x] - 1] = outs[x, -1, 0]
        betta[x, -1, L[x] - 2] = outs[x, -1, labels[x, -1]]
        # betta[:, -1, target_lens[0]-1] = torch.stack([outs[x, -1, 0] for x in range(batch_size)])
        # betta[:, -1, target_lens[0]-2] = torch.stack([outs[x, -1, labels[x, -1]] for x in range(batch_size)])

        d = betta[x, -1, L[x] - 1] + betta[x, -1, L[x] - 2]

        if not torch.equal(d, torch.tensor(0.)):
            betta[x, -1] /= d

    for t in range(T - 1)[::-1]:
        for sample in range(batch_size):
            ss = max(0, L[sample] - 2 * (T - t))
            e = min(2 * t + 2, L[sample])

            for s in range(e)[::-1]:
                i = (s - 1) // 2
                red = betta[sample, t + 1, s]
                blue = 0
                if s < L[sample] - 1:
                    blue = betta[sample, t + 1, s + 1]
                if s % 2 == 0:
                    betta[sample, t, s] = (red + blue) * outs[sample, t, 0]
                elif s == L[sample] - 2 or labels[sample, i] == labels[sample, i + 1]:
                    betta[sample, t, s] = (red + blue) * outs[sample, t, labels[sample, i]]
                else:
                    orange = betta[sample, t + 1, s + 2]
                    betta[sample, t, s] = (red + blue + orange) * outs[sample, t, labels[sample, i]]

            d = torch.sum(betta[sample, t, ss:e], dim=-1)
            if not torch.equal(d, torch.tensor(0.)):
                betta[sample, t, ss:e] /= (d)

    return betta


# works but becomes zero as starts predicting empty words
def find_beta_with_blanks(outs, labels, target_lens):
    batch_size = outs.shape[0]
    T = outs.shape[1]
    L = 2 * target_lens + 1

    betta = torch.zeros((batch_size, T, labels.shape[1]), requires_grad=False, device=torch.device(outs.device))

    for sample in range(batch_size):
        betta[sample, -1, L[sample] - 1] = outs[sample, -1, labels[sample, L[sample] - 1]]
        betta[sample, -1, L[sample] - 2] = outs[sample, -1, labels[sample, L[sample] - 2]]
        d = betta[sample, -1, L[sample] - 1] + betta[sample, -1, L[sample] - 2]
        d = d if torch.is_nonzero(d) else torch.tensor(1.)
        betta[sample, -1, :] /= d

    for sample in range(batch_size):
        for t in range(T - 1)[::-1]:
            ss = max(0, L[sample] - 2 * (T - t))
            e = min(2 * t + 2, L[sample])

            for s in range(e)[::-1]:
                red = betta[sample, t + 1, s]
                blue = betta[sample, t + 1, s + 1] if s < L[sample] - 1 else 0

                if s % 2 == 0 or s == L[sample] - 2 or labels[sample, s] == labels[sample, s + 2]:
                    betta[sample, t, s] = (red + blue) * outs[sample, t, labels[sample, s]]
                else:
                    orange = betta[sample, t + 1, s + 2]
                    betta[sample, t, s] = (red + blue + orange) * outs[sample, t, labels[sample, s]]

            d = torch.sum(betta[sample, t, ss:e], dim=-1)
            d = d if torch.is_nonzero(d) else torch.tensor(1.)
            betta[sample, t, :] /= d

    return betta


# works but becomes zero as starts predicting empty words
def compute_grad_with_blanks(outs, labels, a, b, target_lens):
    L = 2 * target_lens + 1
    T = outs.shape[1]
    batch_size = outs.shape[0]
    k = outs.shape[-1]

    ab = a * b
    grads = torch.zeros((batch_size, T, k), device=torch.device(outs.device))

    for sample in range(batch_size):
        for s in range(L[sample]):
            for t in range(T):
                grads[sample, t, labels[sample, s]] += ab[sample, t, s]
                ab[sample, t, s] /= (outs[sample, t, labels[sample, s]])

    absum = torch.sum(ab, dim=-1)

    for t in range(T):
        for i in range(k):
            grads[:, t, i] = outs[:, t, i] - grads[:, t, i] / (outs[:, t, i] * absum[:, t])  # + eps)

    grads[torch.isnan(grads)] = 0.

    return grads


# doesn't work
def compute_grad(outs, labels, a, b, target_lens):
    L = 2 * target_lens + 1
    T = outs.shape[1]
    batch_size = outs.shape[0]
    k = outs.shape[-1]

    ab = a * b
    grads = torch.zeros((batch_size, T, k), device=torch.device(outs.device))

    for sample in range(batch_size):
        for s in range(L[sample]):
            if s % 2 == 0:
                for t in range(T):
                    grads[sample, t, 0] += ab[sample, t, s]
                    ab[sample, t, s] /= (outs[sample, t, 0])
            else:
                for t in range(T):
                    i = (s - 1) // 2
                    for sample in range(batch_size):
                        grads[sample, t, labels[sample, i]] += ab[sample, t, s]
                        ab[sample, t, s] /= (outs[sample, t, labels[sample, i]])

    absum = torch.sum(ab, dim=-1)

    for t in range(T):
        for i in range(k):
            grads[:, t, i] = outs[:, t, i] - grads[:, t, i] / (outs[:, t, i] * absum[:, t])  # + eps)

    grads[torch.isnan(grads)] = 0.
    return grads
