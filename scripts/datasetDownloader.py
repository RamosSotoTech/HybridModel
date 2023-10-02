import tkinter as tk
from tkinter import ttk
from huggingface_hub import HfApi, DatasetSearchArguments, DatasetFilter
import datasets

api = HfApi()
args = DatasetSearchArguments()

root = tk.Tk()
root.title('Dataset Downloader')

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

dataset_label = ttk.Label(frame, text="Dataset ID:")
dataset_label.grid(row=0, column=0, sticky=tk.W)

license_label = ttk.Label(frame, text="License:")
license_label.grid(row=1, column=0, sticky=tk.W)

config_value = tk.StringVar()
config_combo = ttk.Combobox(frame, textvariable=config_value)
config_combo.grid(row=2, column=0, sticky=tk.W)

filt = DatasetFilter(task_categories=args.task_categories.summarization, language='en', multilinguality="monolingual",
                     size_categories="100K<n<1M")
my_list = api.list_datasets(filter=filt)


def load_next_dataset():
    try:
        dataset = next(my_list)
    except StopIteration:
        print("No more datasets.")
        return

    dataset_id = dataset.id
    try:
        config_options = datasets.get_dataset_config_names(dataset_id, download_mode='reuse_cache_if_exists')
        license_type = get_tag_value("license", dataset)
    except OverflowError:
        print(f"Skipping {dataset_id} due to OverflowError.")
        load_next_dataset()
        return
    except Exception as e:
        print(f"Skipping {dataset_id} due to {e}.")
        load_next_dataset()
        return

    dataset_label.config(text=f"Dataset ID: {dataset_id}")
    license_label.config(text=f"License: {license_type}")
    config_combo['values'] = config_options
    config_value.set(config_options[0] if config_options else "")


def get_tag_value(tag_name, dataset):
    # Initialize an empty list to store the tag values
    tag_values = []

    # Loop through the tags of each item
    for tag in dataset.tags:
        # Split the tag into its name and value
        tag_parts = tag.split(":")

        # Check if the tag name matches the one you are interested in
        if tag_parts[0] == tag_name:
            # Append the value part of the tag to the list
            tag_values.append(tag_parts[1])

    return tag_values


def on_download_click():
    dataset_id = dataset_label.cget("text").split(": ")[1]
    config_selected = config_value.get()
    license_type = license_label.cget("text").split(": ")[1]
    print(f"Downloading {dataset_id} with config {config_selected} and license {license_type}")
    load_next_dataset()


def on_skip_click():
    dataset_id = dataset_label.cget("text").split(": ")[1]
    print(f"Skipping {dataset_id}")
    load_next_dataset()


ttk.Button(frame, text="Download", command=on_download_click).grid(row=3, column=0)
ttk.Button(frame, text="Skip", command=on_skip_click).grid(row=3, column=1)

# Load the first dataset
load_next_dataset()

root.mainloop()
