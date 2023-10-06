import tensorflow as tf
from transformers import TFBertModel, TFGPT2Model, BertTokenizer, GPT2Tokenizer
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from rouge import Rouge
import pickle


# Define the CustomLayer

class HybridModel(tf.keras.Model):
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 gpt2_model_name='gpt2',
                 learning_rate=1e-3,
                 dropout_rate=0.1,
                 regularization_factor=0.01,
                 loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
        super(HybridModel, self).__init__()
        self.bert_encoder = TFBertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_hidden_dim = self.bert_encoder.config.hidden_size
        self.gpt2_decoder = TFGPT2Model.from_pretrained(gpt2_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        gpt2_hidden_dim = self.gpt2_decoder.config.n_embd
        # Initialize intermediate dense layer with L2 regularization
        self.intermediate = tf.keras.layers.Dense(gpt2_hidden_dim,
                                                  kernel_regularizer=l2(regularization_factor))
        self.loss_fn = loss_function
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        self.eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        self.writer = tf.summary.create_file_writer('logs/')
        self.learning_rate = learning_rate

    def call(self, input_segments, sequence_lengths, attention_masks=None, training=False):
        if attention_masks is not None:
            if not tf.is_tensor(attention_masks):
                raise ValueError("attention_masks must be a tensor.")
        # Verify input tensor dimensions
        if len(input_segments.shape) != 2:
            raise ValueError("Expected input_segments to have 2 dimensions, got {}".format(input_segments.shape))

        # Initialize list to store segment representations
        segment_representations = []

        # Loop over each segment in the batch
        for i in range(input_segments.shape[0]):
            input_ids = input_segments[i, :sequence_lengths[i]]
            attention_mask = attention_masks[i, :sequence_lengths[i]] if attention_masks is not None else None

            # Obtain BERT representation for each segment
            bert_output = self.bert_encoder(input_ids, attention_mask=attention_mask)[0]
            segment_representations.append(bert_output[:, 0, :])

        # Step 2: Aggregation Layer
        # Implement a simple attention mechanism
        attention_weights = tf.keras.layers.Attention()([segment_representations, segment_representations])
        document_representation = tf.reduce_sum(attention_weights * segment_representations, axis=1)

        # Feed through intermediate layers
        intermediate_output = self.intermediate(document_representation)
        intermediate_output = self.batch_norm(intermediate_output, training)

        # Step 3: Decoding
        gpt2_outputs = self.gpt2_decoder(inputs_embeds=intermediate_output)[0]

        return gpt2_outputs



    # def call(self, input_segments, attention_masks=None, training=False):
    #     print(f"Debug: input_segments shape: {tf.shape(input_segments)}")
    #     print(f"Debug: attention_masks shape: {tf.shape(attention_masks)}")
    #     input_segments = tf.stack(input_segments)
    #     if attention_masks is not None:
    #         attention_masks = tf.stack(attention_masks)
    #
    #     # Step 1: Sentence/Paragraph Encoding
    #     segment_representations = []
    #     for i, input_ids in enumerate(input_segments):
    #         attention_mask = attention_masks[i] if attention_masks else None
    #
    #         print(f"Debug: input_segments shape: {tf.shape(input_segments)}")
    #         print(f"Debug: attention_masks shape: {tf.shape(attention_masks)}")
    #
    #         bert_output = self.bert_encoder(input_ids, attention_mask=attention_mask)[0]
    #         segment_representations.append(bert_output[:, 0, :])
    #
    #     # Step 2: Aggregation Layer
    #     segment_representations = tf.stack(segment_representations,
    #                                        axis=1)  # Shape: [batch_size, num_segments, hidden_dim]
    #     # Implement a simple attention mechanism
    #     attention_weights = tf.keras.layers.Attention()([segment_representations, segment_representations])
    #     document_representation = tf.reduce_sum(attention_weights * segment_representations, axis=1)
    #
    #     # Feed through intermediate layers
    #     intermediate_output = self.intermediate(document_representation)
    #     intermediate_output = self.batch_norm(intermediate_output, training)
    #
    #     # Step 3: Decoding
    #     gpt2_outputs = self.gpt2_decoder(inputs_embeds=intermediate_output)[0]
    #     return gpt2_outputs

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_ids = batch['target_ids']
        output = self(input_ids, attention_mask)
        target_ids = tf.reshape(target_ids, [-1])
        output = tf.reshape(output, [-1, output.shape[-1]])
        loss = self.loss_fn(target_ids, output)
        self.train_loss_metric.update_state(loss)
        with self.writer.as_default():
            tf.summary.scalar('train_loss', self.train_loss_metric.result(), step=batch['step'])
        return {'loss': loss}

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_ids = batch['target_ids']
        output = self(input_ids, attention_mask)
        target_ids = tf.reshape(target_ids, [-1])
        output = tf.reshape(output, [-1, output.shape[-1]])
        val_loss = self.loss_fn(target_ids, output)
        self.val_loss_metric.update_state(val_loss)
        with self.writer.as_default():
            tf.summary.scalar('val_loss', self.val_loss_metric.result(), step=batch['step'])
        return {'val_loss': val_loss}

    def evaluation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_ids = batch['target_ids']
        output = self(input_ids, attention_mask)
        target_ids = tf.reshape(target_ids, [-1])
        output = tf.reshape(output, [-1, output.shape[-1]])
        eval_loss = self.loss_fn(target_ids, output)
        predicted_summary = self.gpt2_tokenizer.decode(tf.argmax(output, axis=-1), skip_special_tokens=True)
        target_summary = self.gpt2_tokenizer.decode(target_ids, skip_special_tokens=True)
        rouge = Rouge()
        scores = rouge.get_scores(predicted_summary, target_summary, avg=True)
        with tf.summary.create_file_writer('logs').as_default():
            tf.summary.scalar('val_loss', self.val_loss_metric.result(), step=batch['step'])
            tf.summary.scalar('ROUGE-1-score', scores['rouge-1']['f'], step=batch['step'])
            tf.summary.scalar('ROUGE-2-score', scores['rouge-2']['f'], step=batch['step'])
            tf.summary.scalar('ROUGE-L-score', scores['rouge-l']['f'], step=batch['step'])
        return {'eval_loss': eval_loss, 'rouge_scores': scores}

    def save_checkpoint(self, filepath, optimizer):
        self.save_weights(filepath)
        with open(filepath + '_optimizer.pkl', 'wb') as f:
            pickle.dump(optimizer.get_weights(), f)

    @classmethod
    def load_checkpoint(cls, filepath, optimizer):
        model = cls()
        model.load_weights(filepath)
        with open(filepath + '_optimizer.pkl', 'rb') as f:
            optimizer_weights = pickle.load(f)
        optimizer.set_weights(optimizer_weights)
        return model, optimizer
