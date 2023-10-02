import datasets
from datasets import load_dataset
from datasets import list_datasets

# cnn_dataset = load_dataset("cnn_dailymail", "3.0.0")
# cnn_dataset.save_to_disk("SummarizationDataset/cnn_dailymail")

from huggingface_hub import HfApi, DatasetSearchArguments, DatasetFilter

api = HfApi()

# List all SummarizationDataset
api.list_datasets()

# Get all valid search arguments
args = DatasetSearchArguments()

# List only the text classification SummarizationDataset
api.list_datasets(filter="task_categories:text-classification")
# Using the `DatasetFilter`
filt = DatasetFilter(task_categories="text-classification")
# With `DatasetSearchArguments`
filt = DatasetFilter(task=args.task_categories.text_classification)
api.list_models(filter=filt)

# List only the SummarizationDataset in russian for language modeling
api.list_datasets(
    filter=("languages:ru", "task_ids:language-modeling")
)
# Using the `DatasetFilter`
filt = DatasetFilter(languages="ru", task_ids="language-modeling")
# With `DatasetSearchArguments`
filt = DatasetFilter(
    languages=args.languages.ru,
    task_ids=args.task_ids.language_modeling,
)
api.list_datasets(filter=filt)