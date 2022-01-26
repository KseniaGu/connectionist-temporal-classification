# from https://github.com/TheAlgorithms/Python/blob/master/strings/levenshtein_distance.py
def levenshtein_distance(first_word: str, second_word: str) -> int:
    if len(first_word) < len(second_word):
        return levenshtein_distance(second_word, first_word)

    if len(second_word) == 0:
        return len(first_word)

    previous_row = range(len(second_word) + 1)

    for i, c1 in enumerate(first_word):

        current_row = [i + 1]

        for j, c2 in enumerate(second_word):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            # Get the minimum to append to the current row
            current_row.append(min(insertions, deletions, substitutions))

        # Store the previous row
        previous_row = current_row

    # Returns the last element (distance)
    return previous_row[-1]


def decoded_labels(outputs):
    alphabet = '-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    predicted_labels = []

    for encoded_label in outputs:
        raw_label = ''.join([alphabet[encoded_label[i]] for i in range(len(encoded_label))])
        raw_label = ''.join([''.join(dict.fromkeys(words)) for words in raw_label.split('-')])
        predicted_labels.append(raw_label)

    return predicted_labels


def calc_acc(outputs, labels):
    outputs = decoded_labels(outputs)
    labels = decoded_labels(labels)
    nrof_correct = 0

    for label, output in zip(labels, outputs):
        word_0, word_1 = (label, output) if len(label) > len(output) else (output, label)
        lev_distance = levenshtein_distance(word_0, word_1)
        lev_distance /= len(word_0)
        nrof_correct += (1 - lev_distance)

    return nrof_correct


if __name__ == '__main__':
    raw_label = '-mma-aaa--ss-at--'
    print([''.join([''.join(dict.fromkeys(words)) for words in raw_label.split('-')])])
