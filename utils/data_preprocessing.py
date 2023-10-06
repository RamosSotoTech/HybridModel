import os
import pathlib
import pickle
import re
from typing import List, Dict, Tuple

import datasets
import numpy as np
import tensorflow as tf
from datasets import DatasetDict
from datasets import concatenate_datasets, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizer
import h5py

# cnn_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")
# cnn_dataset.save_to_disk("/Users/PC/PycharmProjects/TLDR/datasets/cnn_dailymail")


def load_datasets(path):
    dataset_dict = DatasetDict.load_from_disk(path)
    return dataset_dict['train'], dataset_dict['test'], dataset_dict['validation']


def initialize_tokenizers():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if not bert_tokenizer.pad_token:
        bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if not gpt2_tokenizer.pad_token:
        gpt2_tokenizer.add_special_tokens({'pad_token': ''})  # Using a unique token for potential conflict avoidance

    return bert_tokenizer, gpt2_tokenizer


def is_tqdm(iterable):
    """Check if iterable is an instance of tqdm"""
    return isinstance(iterable, tqdm)


def preprocess_dataset(dataset: Dataset, bert_tokenizer: PreTrainedTokenizer,
                       gpt2_tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Tokenizes the dataset using hierarchical tokenization with both BERT and GPT-2 tokenizers.

    Parameters:
    - dataset (datasets.arrow_dataset.Dataset): Input dataset to tokenize.
    - bert_tokenizer (transformers.PreTrainedTokenizer): BERT tokenizer instance.
    - gpt2_tokenizer (transformers.PreTrainedTokenizer): GPT-2 tokenizer instance.

    Returns:
    - datasets.arrow_dataset.Dataset: Tokenized dataset.
    """

    assert isinstance(dataset, Dataset), f"Expected datasets.arrow_dataset.Dataset but received {type(dataset)}"

    # def tokenization_function(entry):
    #     return hierarchical_tokenization(entry, bert_tokenizer=bert_tokenizer, gpt2_tokenizer=gpt2_tokenizer)

    tokenized_dataset = hierarchical_tokenization(dataset, bert_tokenizer=bert_tokenizer, gpt2_tokenizer=gpt2_tokenizer)#dataset.map(tokenization_function)

    return tokenized_dataset


def create_tf_dataset1(tokenized_data):
    def gen():
        for entry in tokenized_data:
            yield entry['bert_input_segments'], entry['attention_masks'], entry['gpt2_input_ids']

    def map_function(input_segments, attention_masks, gpt2_input_ids):
        return (
            dict(input_segments=input_segments, attention_masks=attention_masks),
            gpt2_input_ids
        )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 512), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, 512), dtype=tf.int32),
            tf.TensorSpec(shape=(512,), dtype=tf.int32)
        )
    )
    dataset = dataset.map(map_function)
    dataset = dataset.batch(8).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_tf_dataset(tokenized_data: Dataset):
    def gen():
        for entry in tokenized_data:
            yield entry['bert_input_segments'], entry['bert_attention_masks'], entry['gpt2_input_ids']

    def map_function(input_segments, attention_masks, gpt2_input_ids):
        return (
            dict(input_segments=input_segments, attention_masks=attention_masks),
            gpt2_input_ids
        )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 512), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, 512), dtype=tf.int32),
            tf.TensorSpec(shape=(512,), dtype=tf.int32)
        )
    )
    dataset = dataset.map(map_function)
    dataset = dataset.batch(8).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


tmp_dir = pathlib.Path("/Users/PC/PycharmProjects/TLDR/tmp")


def split_dataset(deduplicated_data: Dataset, train_ratio: float, val_ratio: float) -> (Dataset, Dataset, Dataset):
    """
    Splits the input dataset into training, validation, and test datasets based on the provided ratios.

    Parameters:
    - deduplicated_data (datasets.arrow_dataset.Dataset): Input dataset to split.
    - train_ratio (float): Proportion of data to be used for training.
    - val_ratio (float): Proportion of data to be used for validation.

    Returns:
    - Tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]:
        The training, validation, and test datasets.
    """

    total_size = len(deduplicated_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data_raw = deduplicated_data.select(list(range(train_size)))
    val_data_raw = deduplicated_data.select(list(range(train_size, train_size + val_size)))
    test_data_raw = deduplicated_data.select(list(range(train_size + val_size, total_size)))

    return train_data_raw, val_data_raw, test_data_raw


def main_pipeline(path='/Users/PC/PycharmProjects/TLDR/datasets/cnn_dailymail', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum up to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios should sum up to 1."

    # 1. Load datasets
    print("Loading datasets...")
    train_data, test_data, val_data = load_datasets(path)

    # Combine datasets into a single dataset
    raw_data = concatenate_datasets(
        [train_data, test_data, val_data])

    # 2. Remove similar articles
    print("Removing similar articles...")
    deduplicated_data = remove_similar_articles(raw_data)

    # 3. Split the data
    train_size = int(len(deduplicated_data) * train_ratio)
    val_size = int(len(deduplicated_data) * val_ratio)

    train_data_raw, val_data_raw, test_data_raw = split_dataset(deduplicated_data, train_ratio, val_ratio)
    # Saving raw data
    with open(os.path.join(tmp_dir, 'train_data_raw.pkl'), 'wb') as f:
        pickle.dump(train_data_raw, f)
    with open(os.path.join(tmp_dir, 'val_data_raw.pkl'), 'wb') as f:
        pickle.dump(val_data_raw, f)
    with open(os.path.join(tmp_dir, 'test_data_raw.pkl'), 'wb') as f:
        pickle.dump(test_data_raw, f)

    # 4. Initialize tokenizers
    print("Initializing tokenizers...")
    bert_tokenizer, gpt2_tokenizer = initialize_tokenizers()

    # 5. Preprocess datasets
    print("Preprocessing train dataset...")
    train_data = preprocess_dataset(train_data_raw, bert_tokenizer, gpt2_tokenizer)
    print("Preprocessing test dataset...")
    test_data = preprocess_dataset(test_data_raw, bert_tokenizer, gpt2_tokenizer)
    print("Preprocessing validation dataset...")
    val_data = preprocess_dataset(val_data_raw, bert_tokenizer, gpt2_tokenizer)

    # Saving tokenized data
    with open(os.path.join(tmp_dir, 'train_data_tokenized.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(tmp_dir, 'val_data_tokenized.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(tmp_dir, 'test_data_tokenized.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # 6. Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_tf_dataset(train_data)
    test_dataset = create_tf_dataset(test_data)
    val_dataset = create_tf_dataset(val_data)

    return train_dataset, test_dataset, val_dataset


def save_to_hdf5(prefetch_dataset, file_path):
    with h5py.File(file_path, 'w') as hf:
        for i, (input_dict, labels) in enumerate(prefetch_dataset):
            grp = hf.create_group(str(i))

            # Save input dictionary
            input_grp = grp.create_group('input')
            for key, value in input_dict.items():
                input_grp.create_dataset(key, data=value.numpy())

            # Save labels
            grp.create_dataset('labels', data=labels.numpy())


def load_raw_data(save_dir='saved_data'):
    with open(os.path.join(save_dir, 'train_data_raw.pkl'), 'rb') as f:
        train_data_raw = pickle.load(f)
    with open(os.path.join(save_dir, 'val_data_raw.pkl'), 'rb') as f:
        val_data_raw = pickle.load(f)
    with open(os.path.join(save_dir, 'test_data_raw.pkl'), 'rb') as f:
        test_data_raw = pickle.load(f)

    return train_data_raw, val_data_raw, test_data_raw


def load_tokenized_data(save_dir='saved_data'):
    with open(os.path.join(save_dir, 'train_data_tokenized.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(save_dir, 'val_data_tokenized.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(save_dir, 'test_data_tokenized.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(example):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Here, the structure of the dataset needs to be reflected.
    # The below structure assumes a simple example dictionary with 'id', 'bert_input_segments', etc.
    # You might need to adjust depending on the exact structure of your data.
    feature = {
        'id': _bytes_feature(example['id'].encode('utf-8')),
        # Assuming 'bert_input_segments' and others are lists of numpy arrays:
        'bert_input_segments': _bytes_feature(tf.io.serialize_tensor(example['bert_input_segments'])),
        'bert_attention_masks': _bytes_feature(tf.io.serialize_tensor(example['bert_attention_masks'])),
        'gpt2_input_ids': _bytes_feature(tf.io.serialize_tensor(example['gpt2_input_ids']))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(data, filename):
    def gen():
        for entry in data:
            if 'attention_masks' not in entry:
                print(f"Faulty entry keys: {entry.keys()}")
                continue  # skip this entry and move to the next one
            yield entry['bert_input_segments'], entry['attention_masks'], entry['gpt2_input_ids']

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    with tf.io.TFRecordWriter(filename) as writer:
        for batch in dataset:
            for item in batch:
                serialized_example = serialize_example(item)
                writer.write(serialized_example)


def convert_to_numpy(data):
    """
    Converts any TensorFlow tensors within the data to numpy arrays or string for non-directly convertible tensors.
    """
    if isinstance(data, tf.Tensor):
        if data.dtype == tf.variant:
            # Convert to a string and then to numpy
            return np.array(str(data))
        else:
            return data.numpy()
    elif isinstance(data, dict):
        # Ensure the dictionary has the expected keys and handle them explicitly
        assert all(key in data for key in
                   ['bert_input_segments', 'attention_masks', 'gpt2_input_ids']), "Unexpected dictionary structure"
        return {
            'bert_input_segments': convert_to_numpy(data['bert_input_segments']),
            'attention_masks': convert_to_numpy(data['attention_masks']),
            'gpt2_input_ids': convert_to_numpy(data['gpt2_input_ids']),
        }
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    else:
        return data


def process_and_save_data(data_path: str):
    """
    Processes data using main_pipeline and then saves the tokenized train, validation,
    and test data in separate files inside a directory named 'tmp'.

    Parameters:
    - data_path: str, path to the dataset

    Returns:
    None
    """

    train_data, validation_data, test_data = main_pipeline(data_path)

    # Convert TensorFlow tensors to numpy arrays
    train_data = convert_to_numpy(train_data)
    for entry in train_data:
        assert all(k in entry for k in
                   ['bert_input_segments', 'attention_masks', 'gpt2_input_ids']), f"Entry missing keys: {entry.keys()}"
    validation_data = convert_to_numpy(validation_data)
    test_data = convert_to_numpy(test_data)

    # Ensure the 'tmp' directory exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    tmp_dir = "/Users/PC/PycharmProjects/TLDR/tmp"
    # Save processed data into separate files
    write_tfrecords(train_data, os.path.join(tmp_dir, 'train_data.tfrecord'))
    write_tfrecords(validation_data, os.path.join(tmp_dir, 'val_data.tfrecord'))
    write_tfrecords(test_data, os.path.join(tmp_dir, 'test_data.tfrecord'))

    print("Processed data has been saved in the 'tmp' directory.")


def get_article_and_highlights_by_id(dataset_dict, target_id):
    for split, dataset in dataset_dict.items():
        for x in dataset:
            article_id = x.get('id', None)  # Assuming the ID field is named 'id'
            if article_id == target_id:
                return {
                    'split': split,
                    'article': x['article'],
                    'highlights': x['highlights']
                }
    return None  # Return None if the ID is not found in any split


def remove_similar_articles(dataset: datasets.Dataset, stop_words: set = None, threshold: float = 0.9,
                            batch_size: int = 500) -> datasets.Dataset:
    """
    Remove articles from the dataset that are similar based on the provided threshold.

    Parameters:
    - dataset (datasets.arrow_dataset.Dataset): Input dataset containing articles.
    - stop_words (set, optional): Set of words to exclude during vectorization. Defaults to None.
    - threshold (float, optional): Cosine similarity threshold above which articles are considered similar. Defaults to 0.9.
    - batch_size (int, optional): Number of articles to process in each batch. Defaults to 500.

    Returns:
    - datasets.arrow_dataset.Dataset: Dataset with similar articles removed.
    """

    # Vectorize articles
    vectorizer = TfidfVectorizer(stop_words=list(stop_words) if stop_words else None)
    tfidf_matrix = vectorizer.fit_transform([entry['article'] for entry in dataset])
    num_articles = tfidf_matrix.shape[0]

    indices_to_remove = set()

    for start_idx in tqdm(range(0, num_articles, batch_size), desc='Checking for similar articles'):
        end_idx = min(start_idx + batch_size, num_articles)
        cosine_matrix = cosine_similarity(tfidf_matrix[start_idx:end_idx], tfidf_matrix)

        # Identify similar articles based on threshold
        for i, row in enumerate(cosine_matrix):
            similar_indices = set(np.where(row > threshold)[0])
            similar_indices.discard(start_idx + i)  # Remove self-index
            indices_to_remove.update(similar_indices)

    # Filter out articles with indices in indices_to_remove
    filtered_dataset = dataset.filter(lambda example, idx: idx not in indices_to_remove, with_indices=True)

    return filtered_dataset


def clean_text(text: str) -> str:
    return remove_extra_spaces(re.sub(r'\n', ' ', text))


def remove_extra_spaces(sentence: str) -> str:
    # Use regex to replace multiple spaces with a single space
    cleaned_sentence = re.sub(r'\s+', ' ', sentence).strip()
    return cleaned_sentence


def replace_urls(text: str, replacement: str = '[URL]') -> str:
    # Regular expression pattern for detecting URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Replace detected URLs with the desired replacement string
    cleaned_text = url_pattern.sub(replacement, text)

    return cleaned_text


def custom_preprocessing(dataset: List[Dict], bert_tokenizer: BertTokenizer, gpt2_tokenizer: GPT2Tokenizer,
                         max_length: int = 512) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """
    Custom preprocessing function for text data.

    Parameters:
    - dataset (List[Dict]): The input data to be preprocessed.
    - bert_tokenizer (BertTokenizer): The tokenizer instance for BERT.
    - gpt2_tokenizer (GPT2Tokenizer): The tokenizer instance for GPT-2.
    - max_length (int, optional): The maximum sequence length for BERT chunks. Default is 512.

    Returns:
    - List[tf.Tensor]: List of tensors of bert_input_ids for each chunk.
    - List[tf.Tensor]: List of tensors of bert_attention_masks for each chunk.
    - List[tf.Tensor]: Tensor of gpt2_input_ids for the highlights/summaries.
    """

    bert_input_ids_list = []
    bert_attention_masks_list = []
    gpt2_input_ids_list = []

    for entry in dataset:
        # BERT Tokenization for entire article
        encoded_article = bert_tokenizer.encode_plus(
            entry['article'],
            add_special_tokens=True,
            max_length=None,
            truncation=False,
            return_attention_mask=True,
            return_tensors="tf"  # Return TensorFlow tensors
        )

        # Chunking BERT tokens
        input_ids = encoded_article['input_ids'][0]
        attention_mask = encoded_article['attention_mask'][0]

        bert_input_ids = [input_ids[i:i + max_length].numpy() for i in range(0, len(input_ids), max_length - 1)]
        bert_attention_masks = [attention_mask[i:i + max_length].numpy() for i in
                                range(0, len(attention_mask), max_length - 1)]

        bert_input_ids_list.extend(bert_input_ids)
        bert_attention_masks_list.extend(bert_attention_masks)

        # GPT-2 Tokenization for the highlights/summaries
        gpt2_tokens = gpt2_tokenizer.encode_plus(
            entry['highlights'],  # Assuming 'highlights' or 'summary' field is present
            add_special_tokens=True,
            return_tensors="tf"  # Return TensorFlow tensors
        )

        gpt2_input_ids_list.append(gpt2_tokens['input_ids'])

    return bert_input_ids_list, bert_attention_masks_list, gpt2_input_ids_list



def hierarchical_tokenization1(dataset: Dataset, bert_tokenizer, gpt2_tokenizer) -> Dataset:
    """
    Conducts hierarchical tokenization for a hybrid model.

    Parameters:
    - dataset: datasets.arrow_dataset.Dataset object, each entry containing 'id', 'article', and 'highlights'
    - bert_tokenizer: BertTokenizer, specifies the pretrained BERT model to use
    - gpt2_tokenizer: GPT2Tokenizer, specifies the pretrained GPT-2 model to use

    Returns:
    - datasets.arrow_dataset.Dataset: tokenized_data
    """
    max_length = 512  # BERT's maximum token length

    def tokenize_entry(entry: Dict):
        article = clean_text(entry['article'])
        highlights = clean_text(entry['highlights'])

        # BERT tokenization for the entire article
        encoded_article = bert_tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="np",
            max_length=None,
            truncation=False
        )

        # Divide article into chunks
        input_ids = encoded_article['input_ids'][0]
        attention_mask = encoded_article['attention_mask'][0]

        segment_input_ids = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length - 1)]
        segment_attention_masks = [attention_mask[i:i + max_length] for i in
                                   range(0, len(attention_mask), max_length - 1)]

        # GPT-2 tokenization for the highlights
        gpt2_tokenized_highlights = gpt2_tokenizer.encode_plus(
            highlights,
            add_special_tokens=True,
            return_tensors="np"
        )

        # Create entry for tokenized data
        return {
            'id': entry['id'],
            'bert_input_segments': segment_input_ids,
            'bert_attention_masks': segment_attention_masks,
            'gpt2_input_ids': gpt2_tokenized_highlights['input_ids']
        }

    # Tokenize all entries in the dataset
    tokenized_data = dataset.map(tokenize_entry)

    return tokenized_data


def hierarchical_tokenization(dataset: Dataset, bert_tokenizer, gpt2_tokenizer) -> Dataset:
    """
    Conducts hierarchical tokenization for a hybrid model.

    Parameters:
    - dataset: datasets.arrow_dataset.Dataset object, each entry containing 'id', 'article', and 'highlights'
    - bert_tokenizer: BertTokenizer, specifies the pretrained BERT model to use
    - gpt2_tokenizer: GPT2Tokenizer, specifies the pretrained GPT-2 model to use

    Returns:
    - datasets.arrow_dataset.Dataset: tokenized_data
    """
    max_length = 512  # BERT's maximum token length

    def tokenize_entry(entry: Dict):
        article = clean_text(entry['article'])
        highlights = clean_text(entry['highlights'])

        # BERT tokenization for the entire article
        encoded_article = bert_tokenizer.encode_plus(
            article,
            add_special_tokens=False,  # Handle special tokens manually for segments
            return_attention_mask=True,
            return_tensors="tf",
            max_length=None,
            truncation=False
        )

        # Divide article into chunks
        input_ids = encoded_article['input_ids'][0]
        attention_mask = encoded_article['attention_mask'][0]

        segment_input_ids = [tf.concat([[101], input_ids[i:i + max_length - 2], [102]], axis=0) for i in
                             range(0, len(input_ids), max_length - 2)]
        segment_attention_masks = [tf.concat([[1], attention_mask[i:i + max_length - 2], [1]], axis=0) for i in
                                   range(0, len(attention_mask), max_length - 2)]

        # Pad if necessary
        segment_input_ids = [tf.pad(x, paddings=[[0, max_length - tf.shape(x)[0]]], mode='CONSTANT', constant_values=0)
                             for x in segment_input_ids]
        segment_attention_masks = [
            tf.pad(x, paddings=[[0, max_length - tf.shape(x)[0]]], mode='CONSTANT', constant_values=0) for x in
            segment_attention_masks]

        # GPT-2 tokenization for the highlights
        gpt2_tokenized_highlights = gpt2_tokenizer.encode_plus(
            highlights,
            add_special_tokens=True,
            return_tensors="tf"
        )

        # Create entry for tokenized data
        return {
            'id': entry['id'],
            'bert_input_segments': tf.stack(segment_input_ids),
            'bert_attention_masks': tf.stack(segment_attention_masks),
            'gpt2_input_ids': gpt2_tokenized_highlights['input_ids'],
            'sequence_lengths': tf.convert_to_tensor([len(x) for x in segment_input_ids])
        }

    # Tokenize all entries in the dataset
    tokenized_data = dataset.map(tokenize_entry)

    return tokenized_data


def hierarchical_tokenization2(dataset: List[Dict], bert_tokenizer: BertTokenizer, gpt2_tokenizer: GPT2Tokenizer) -> \
        List[Dict]:
    """
    Conducts hierarchical tokenization for a hybrid model.

    Parameters:
    - dataset: List of dictionaries, each containing 'id', 'article', and 'highlights'
    - bert_tokenizer: BertTokenizer, specifies the pretrained BERT model to use
    - gpt2_tokenizer: GPT2Tokenizer, specifies the pretrained GPT-2 model to use

    Returns:
    - List[Dict]: tokenized_data
    """
    tokenized_data = []
    max_length = 512  # BERT's maximum token length

    for entry in dataset:
        article = clean_text(entry['article'])
        highlights = clean_text(entry['highlights'])

        # BERT tokenization for the entire article
        encoded_article = bert_tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="np",
            max_length=None,
            truncation=False
        )

        # Divide article into chunks
        input_ids = encoded_article['input_ids'][0]
        attention_mask = encoded_article['attention_mask'][0]

        segment_input_ids = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length - 1)]
        segment_attention_masks = [attention_mask[i:i + max_length] for i in
                                   range(0, len(attention_mask), max_length - 1)]

        # GPT-2 tokenization for the highlights
        gpt2_tokenized_highlights = gpt2_tokenizer.encode_plus(
            highlights,
            add_special_tokens=True,
            return_tensors="np"
        )

        # Create entry for tokenized data
        tokenized_entry = {
            'id': entry['id'],
            'bert_input_segments': segment_input_ids,
            'bert_attention_masks': segment_attention_masks,
            'gpt2_input_ids': gpt2_tokenized_highlights['input_ids']
        }

        tokenized_data.append(tokenized_entry)

    return tokenized_data


def real_time_preprocessing(article, bert_tokenizer, gpt2_tokenizer):
    # Process single article
    tokenized_data = hierarchical_tokenization([article], bert_tokenizer=bert_tokenizer, gpt2_tokenizer=gpt2_tokenizer)
    return tokenized_data


def get_summary(article, model, bert_tokenizer, gpt2_tokenizer):
    tokenized_data = real_time_preprocessing(article, bert_tokenizer, gpt2_tokenizer)
    model_input = {
        "input_segments": tokenized_data['input_segments'],
        "attention_masks": tokenized_data['attention_masks']
    }
    logits = model(model_input)
    predicted_ids = tf.argmax(logits, axis=-1)
    summary = gpt2_tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return summary
