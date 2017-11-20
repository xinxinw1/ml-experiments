import shutil

def group(lst, n):
    return (lst[i:i+n] for i in range(0, len(lst), n))

def flatmap(f, lst):
    return (inner_item for item in lst for inner_item in f(item))

def split_max_len(lst_of_lsts, max_len):
    return list(flatmap(lambda lst: group(lst, max_len), lst_of_lsts))

def pad_to_len(lst, max_len, pad_item):
    return lst + [pad_item] * max(0, max_len-len(lst))

def pad_to_same_len(lst_of_lsts, pad_item):
    max_len = max(map(len, lst_of_lsts))
    return list(map(lambda lst: pad_to_len(lst, max_len, pad_item), lst_of_lsts))

def make_batches(inputs, batch_size, max_single_len, token_item, pad_item):
    inputs_batches = map(lambda input: [token_item] + input, inputs)
    labels_batches = map(lambda input: input + [token_item], inputs)
    if max_single_len is not None:
        inputs_batches = split_max_len(inputs_batches, max_single_len)
        labels_batches = split_max_len(labels_batches, max_single_len)
    inputs_batches = group(list(inputs_batches), batch_size)
    labels_batches = group(list(labels_batches), batch_size)
    inputs_batches = map(lambda batch: pad_to_same_len(batch, pad_item), inputs_batches)
    labels_batches = map(lambda batch: pad_to_same_len(batch, pad_item), labels_batches)

    return zip(inputs_batches, labels_batches)



def rmdir_if_exists(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

