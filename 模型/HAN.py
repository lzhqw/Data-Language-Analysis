import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class HierarchialAttentionNetwork(nn.Module):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, numeric_size,
                 dense_hidden_size, dropout=0.5):
        """
        :param n_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(HierarchialAttentionNetwork, self).__init__()

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, dropout)

        # Classifier
        self.fc1 = nn.Linear(2 * sentence_rnn_size, dense_hidden_size)
        self.fc2 = nn.Linear(numeric_size, dense_hidden_size)
        self.fc3 = nn.Linear(2 * dense_hidden_size, 1)
        self.fc4 = nn.Linear(dense_hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.bn1 = nn.BatchNorm1d(2 * dense_hidden_size)
        self.bn2 = nn.BatchNorm1d(dense_hidden_size)

    def forward(self, documents, sentences_per_document, words_per_sentence, numeric):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        # Apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document,
                                                                                    words_per_sentence)  # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))

        # Classify
        # document_feature = F.relu(self.fc1(self.dropout(document_embeddings)))
        # numeric_feature = F.relu(self.fc2(self.dropout(numeric)))
        # scores = self.fc3(
        #     self.dropout(torch.concat([document_feature, numeric_feature], dim=1)))  # (n_documents, n_classes)
        # scores = torch.sigmoid(scores)
        
        document_feature = self.fc1(document_embeddings)
        document_feature = self.bn2(document_feature)
        document_feature = F.relu(document_feature)
        
        # document_feature = self.dropout(document_feature)

        numeric_feature = self.fc2(numeric)
        numeric_feature = self.bn2(numeric_feature)
        numeric_feature = F.relu(numeric_feature)
        # numeric_feature = self.dropout(numeric_feature)

        feature = torch.concat([document_feature, numeric_feature], dim=1)
        scores = self.fc3(feature)
        # feature = document_feature + numeric_feature
        # scores = self.fc4(feature)
        scores = torch.sigmoid(scores)
        
        return scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,
                                            dropout)

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector
        # You could also do this with:
        # self.sentence_context_vector = nn.Parameter(torch.FloatTensor(1, sentence_att_size))
        # self.sentence_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """

        # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(documents,
                                                lengths=sentences_per_document.tolist(),
                                                batch_first=True,
                                                enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence,
                                                         lengths=sentences_per_document.tolist(),
                                                         batch_first=True,
                                                         enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)
        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(packed_sentences.data,
                                                     packed_words_per_sentence.data)  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)

        # Apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences,
                                                               batch_sizes=packed_sentences.batch_sizes,
                                                               sorted_indices=packed_sentences.sorted_indices,
                                                               unsorted_indices=packed_sentences.unsorted_indices))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_s = self.sentence_attention(packed_sentences.data)  # (n_sentences, att_size)
        att_s = torch.tanh(att_s)  # (n_sentences, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(data=att_s,
                                                      batch_sizes=packed_sentences.batch_sizes,
                                                      sorted_indices=packed_sentences.sorted_indices,
                                                      unsorted_indices=packed_sentences.unsorted_indices),
                                       batch_first=True)  # (n_documents, max(sentences_per_document))

        # Calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)  # (n_documents, max(sentences_per_document))

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(packed_sentences,
                                           batch_first=True)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        # Find document embeddings
        documents = documents * sentence_alphas.unsqueeze(
            2)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)
        documents = documents.sum(dim=1)  # (n_documents, 2 * sentence_rnn_size)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas,
                                                            batch_sizes=packed_sentences.batch_sizes,
                                                            sorted_indices=packed_sentences.sorted_indices,
                                                            unsorted_indices=packed_sentences.unsorted_indices),
                                             batch_first=True)  # (n_documents, max(sentences_per_document), max(words_per_sentence))

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_sentences, word_pad_len, emb_size)
        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(sentences,
                                            lengths=words_per_sentence.tolist(),
                                            batch_first=True,
                                            enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words)  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)

        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas


def train_one_epoch(net, optimizer, loss, train_loader, device):
    metrics = [0, 0, 0]
    net.train()
    t = tqdm(train_loader)
    for batch in t:
        # -------------------------------------------- #
        # 训练
        # -------------------------------------------- #
        documents, sentences_per_document, words_per_sentence, numeric, y = [x.to(device) for x in batch]
        y = torch.unsqueeze(y, dim=1)
        scores, word_alphas, sentence_alphas = net(documents, sentences_per_document, words_per_sentence, numeric)
        l = loss(scores, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # -------------------------------------------- #
        # 计算准确率, loss求和
        # -------------------------------------------- #
        for i in range(y.shape[0]):
            res = 1 if scores[i, 0] >= 0.5 else 0
            if y[i, 0] == res:
                metrics[1] += 1
            metrics[0] += l.item()
            metrics[2] += 1
        t.set_postfix(train_loss=metrics[0] / metrics[2], train_acc=metrics[1] / metrics[2])
    return metrics


def val_one_epoch(net, loss, test_loader, device):
    net.eval()
    metrics = [0, 0, 0]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # -------------------------------------------- #
            # 前向传播
            # -------------------------------------------- #
            documents, sentences_per_document, words_per_sentence, numeric, y = [x.to(device) for x in batch]
            y = torch.unsqueeze(y, dim=1)
            scores, word_alphas, sentence_alphas = net(documents, sentences_per_document, words_per_sentence, numeric)
            l = loss(scores, y)
            # -------------------------------------------- #
            # 计算准确率, loss求和
            # -------------------------------------------- #
            for i in range(y.shape[0]):
                res = 1 if scores[i, 0] >= 0.5 else 0
                if y[i, 0] == res:
                    metrics[1] += 1
                metrics[0] += l.item()
                metrics[2] += 1
    return metrics


def train(net, train_loader_params, test_loader, lr, epochs, device, weight_path, logpath=''):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    # -------------------------------------------- #
    # 网络、优化器、loss
    # -------------------------------------------- #
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.BCELoss()
    # -------------------------------------------- #
    # 记录训练loss acc, 测试 loss acc
    # -------------------------------------------- #
    train_history = []
    test_history = []
    for epoch in range(epochs):
        train_loader = DataLoader(**train_loader_params)
        metrics = train_one_epoch(net=net,
                                  optimizer=optimizer,
                                  loss=loss,
                                  device=device,
                                  train_loader=train_loader)
        # -------------------------------------------- #
        # 平均loss 平均 acc
        # -------------------------------------------- #
        train_loss = metrics[0] / metrics[2]
        train_acc = metrics[1] / metrics[2]
        train_history.append((train_loss, train_acc))
        print(f'epoch:{epoch} train_loss:{train_loss:.3f} train_acc:{train_acc}')
        with open(logpath + 'log_train.txt', mode='a') as f:
            f.write(f'{epoch},{train_loss},{train_acc}\n')
        f.close()

        if epoch % 1 == 0:
            metrics = val_one_epoch(net=net, test_loader=test_loader, loss=loss, device=device)
            # -------------------------------------------- #
            # 平均loss 平均 acc
            # -------------------------------------------- #
            test_loss = metrics[0] / metrics[2]
            test_acc = metrics[1] / metrics[2]
            test_history.append((test_loss, test_acc))
            print(f'epoch:{epoch} test_loss:{test_loss:.3f} test_acc:{test_acc}')
            with open(logpath + 'log_test.txt', mode='a') as f:
                f.write(f'{epoch},{test_loss},{test_acc}\n')
            f.close()
            torch.save(net.state_dict(),
                       '{}/eopch_{}_loss_{:.4f}_acc{}.pth'.format(weight_path, epoch, test_loss, test_acc))
    return train_history, test_history

def visualize_attention(doc, scores, word_alphas, sentence_alphas, words_in_each_sentence, y):
    """
    Visualize important sentences and words, as seen by the HAN model.
    :param doc: pre-processed tokenized document
    :param scores: class scores, a tensor of size (n_classes)
    :param word_alphas: attention weights of words, a tensor of size (n_sentences, max_sent_len_in_document)
    :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)
    """
    # Find best prediction
    # score, prediction = scores.max(dim=0)

    if scores.item() >= 0.5:
        score = scores
        prediction = 'success'
    else:
        score = 1 - scores
        prediction = 'fail'

    print(score.item(), prediction)
    prediction = '{category} ({score:.2f}%)'.format(category=prediction, score=score.item() * 100)

    # For each word, find it's effective importance (sentence alpha * word alpha)
    alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(
        1).float() / words_in_each_sentence.max().float())
    # alphas = word_alphas * words_in_each_sentence.unsqueeze(1).float() / words_in_each_sentence.max().float()
    alphas = alphas.to('cpu')

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word, because size is scaled by normalized word*sentence alphas
    max_font_size = 55  # maximum size possible for a word, because size is scaled by normalized word*sentence alphas
    space_size = ImageFont.truetype("./MS Mincho.ttf", max_font_size).getsize(' ')  # use spaces of maximum font size
    line_spacing = 15  # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    max_rectangle_width = 0.8 * left_buffer  # maximum width of the rectangles that will represent sentence alphas, scaled by sentence alpha
    rectangle_loc = [0.9 * left_buffer,
                     image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = list()
    sentence_viz_properties = list()
    for s, sentence in enumerate(doc):
        # Find visualization properties for each sentence, represented by rectangles
        # Factor to scale by
        sentence_factor = sentence_alphas[s].item() / sentence_alphas.max().item()

        # Color of rectangle
        rectangle_saturation = str(int(sentence_factor * 100))
        rectangle_lightness = str(25 + 50 - int(sentence_factor * 50))
        rectangle_color = 'hsl(0,' + rectangle_saturation + '%,' + rectangle_lightness + '%)'

        # Bounds of rectangle
        rectangle_bounds = [rectangle_loc[0] - sentence_factor * max_rectangle_width,
                            rectangle_loc[1] - rectangle_height] + rectangle_loc

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})

        for w, word in enumerate(sentence):
            # Find visualization properties for each word
            # Factor to scale by
            word_factor = alphas[s, w].item() / alphas.max().item()

            # Color of word
            word_saturation = str(int(word_factor * 100))
            word_lightness = str(25 + 50 - int(word_factor * 50))
            word_color = 'hsl(0,' + word_saturation + '%,' + word_lightness + '%)'

            # Size of word
            word_font_size = int(min_font_size + word_factor * (max_font_size - min_font_size))
            word_font = ImageFont.truetype("./MS Mincho.ttf", word_font_size)

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = word_font.getsize(word)
            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.text(xy=viz['loc'], text=viz['word'], fill=viz['color'], font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])
    # Detected category/topic
    category_font = ImageFont.truetype("./MS Mincho.ttf", min_font_size)
    draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    draw.text(xy=[line_spacing, line_spacing + category_font.getsize('Detected Category:')[1] + line_spacing],
              text=prediction.upper(), fill='black',
              font=category_font)
    del draw

    # Display
    # img.save('img/' + f'{prediction[:-8]} {score.item():.3f} {y} ' + ''.join(
    #     i for i in doc[0] if not i in ['\\', '?', ':', '/', '<', '>', '|', '*', '\"']) + '.jpg', quality=95,
    #          subsampling=0)
    img.show()