import datasets
import functools
import torch
import torchtext
from src.utils import tokenize_data, numericalize_data, collate

def get_data():
    train_data, test_data = datasets.load_dataset('imdb', split=['train', 'test'], cache_dir="data/imdb")
    return train_data, test_data

def get_tokenizer():
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    return tokenizer

def process_data(train_data, test_data, tokenizer):
    max_length = 256

    train_data = train_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
    test_data = test_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

    test_size = 0.25

    train_valid_data = train_data.train_test_split(test_size=test_size)
    train_data = train_valid_data['train']
    valid_data = train_valid_data['test']

    min_freq = 5
    special_tokens = ['<unk>', '<pad>']

    vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                    min_freq=min_freq,
                                                    specials=special_tokens)

    unk_index = vocab['<unk>']

    vocab.set_default_index(unk_index)
    train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

    train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    valid_data = valid_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length'])


    return train_data, valid_data, test_data, vocab


def get_data_loaders(batch_size=512):
    train_data, test_data = get_data()
    tokenizer = get_tokenizer()
    train_data, valid_data, test_data, vocab = process_data(train_data, test_data, tokenizer)

    pad_index = vocab['<pad>']

    collate_fn = functools.partial(collate, pad_index=pad_index)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=batch_size, 
                                                collate_fn=collate_fn, 
                                                shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader, vocab



# print("Running Quick Tests:")
# train_dataloader, valid_dataloader, test_dataloader, vocab = get_data_loaders()

# for batch in train_dataloader:
#     print(f"Feature batch shape: {batch['ids'].size()}")
#     print(f"Labels batch shape: {batch['label'].size()}")

#     print(f"Len batch shape: {batch['length'].size()}")
#     break











