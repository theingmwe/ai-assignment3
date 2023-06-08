# This file implements a Naive Bayes Classifier
import math


class BayesClassifier:
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """

    def __init__(self, alpha=0.1):
        self.positive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_scentences = 0
        self.percent_negative_scentences = 0
        self.alpha = alpha
        self.file_length = 499
        self.file_sections = [
            self.file_length // 4,
            self.file_length // 3,
            self.file_length // 2,
        ]

    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        # Initialize variables to store the number of positive and negative sentences
        self.number_positive_sentences = 0
        self.number_negative_sentences = 0

        # Calculate the percentage of positive and negative sentences
        for label in train_labels:
            if label == "1":
                self.number_positive_sentences += 1
            else:
                self.number_negative_sentences += 1

        total_positives = sum(self.positive_word_counts.values())
        total_negatives = sum(self.negative_word_counts.values())

        self.percent_positive_sentences = self.number_positive_sentences / len(
            train_labels
        )
        self.percent_negative_sentences = self.number_negative_sentences / len(
            train_labels
        )

        # Initialize dictionaries to store the positive and negative word counts
        self.positive_word_counts = {}
        self.negative_word_counts = {}

        # Calculate the word counts for each word in the vocab
        for i, word in enumerate(vocab):
            # Initialize the word counts for the current word with dirichlet prior value
            self.positive_word_counts[word] = self.alpha
            self.negative_word_counts[word] = self.alpha

            # Iterate over the train vectors and labels
            for vector, label in zip(train_data, train_labels):
                # Check if the current word is present in the vector and update the word counts accordingly
                if label == "1" and vector[i] == "1":
                    self.positive_word_counts[word] += 1
                elif vector[i] == "1":
                    self.negative_word_counts[word] += 1

        # Calculate the total number of words with Dirichlet prior
        total_positives += self.alpha * len(vocab)
        total_negatives += self.alpha * len(vocab)

        # Update the percentage of positive and negative sentences
        self.percent_positive_sentences = self.number_positive_sentences / len(train_labels)
        self.percent_negative_sentences = self.number_negative_sentences / len(train_labels)

        # Update the word probabilities with Dirichlet prior
        for word in self.positive_word_counts:
            self.positive_word_counts[word] /= total_positives
        for word in self.negative_word_counts:
            self.negative_word_counts[word] /= total_negatives

        return 1

    def classify_text(self, vectors, vocab):
        """
        Classifies the given text vectors
        vectors: list of vectorized text
        vocab: vocab from build_vocab
        Returns a list of predictions
        """
        # Initialize an empty list to store the predictions
        predictions = []

        # Iterate over each vector in the input vectors
        for vector in vectors:
            # Calculate the log probabilities of positive and negative sentences
            percent_pos = math.log(max(self.percent_positive_sentences, 1))
            percent_neg = math.log(max(self.percent_negative_sentences, 1))

            # Iterate over each element in the vector and the corresponding word in the vocab
            for seen, word in zip(vector, vocab):
                if word in self.positive_word_counts and word in self.negative_word_counts:
                    # Check if the current word is present in the vector and update the log probabilities accordingly
                    if seen == "1":
                        percent_pos += math.log(
                            (self.positive_word_counts[word])
                            / (self.number_positive_sentences + len(vocab))
                        )
                        percent_neg += math.log(
                            (self.negative_word_counts[word])
                            / (self.number_negative_sentences + len(vocab))
                        )

            # Make a prediction based on the calculated log probabilities
            if percent_pos > percent_neg:
                predictions.append("1")
            else:
                predictions.append("0")

        # Return the list of predictions
        return predictions
