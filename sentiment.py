# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions

import string

from classifier import BayesClassifier


def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    # Remove apostrophes
    text = text.replace("'", "")

    # Remove punctuation marksss
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # Split the text into a list of words and make it lowercase
    preprocessed_text = text.lower().split()

    return preprocessed_text


def build_vocab(preprocessed_text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    # Initialize an empty set to store the unique text tokens
    vocab = set()

    # Iterate over each sentence in the preprocessed text
    for sentence in preprocessed_text:
        # Iterate over each word in the sentence
        if sentence not in vocab and not any(char.isdigit() for char in sentence):
            vocab.add(sentence)

    # Sort the vocab
    vocab = sorted(vocab)

    # Return the vocab set containing the unique text tokens
    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """

    # Initialize an empty list to store the vectorized text
    vectorized_text = []

    # Iterate over each word in the vocab
    for word in vocab:
        # Check if the word is in the text
        if word in text:
            # Append '1' to the vectorized text if the word is in the text
            vectorized_text.append("1")
        else:
            # Append '0' to the vectorized text if the word is not in the text
            vectorized_text.append("0")

    # Get the label from the last element of the text input
    label = text[-1]

    # Return the vectorized text and label
    return vectorized_text, label


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    # Initialize a variable to store the number of correct predictions
    correct_predictions = 0

    # Iterate over the predicted and true labels
    for predicted, true in zip(predicted_labels, true_labels):
        # Check if the predicted label matches the true label
        if predicted == true:
            # Increment the number of correct predictions
            correct_predictions += 1

    # Calculate the accuracy as the ratio of correct predictions to total predictions
    accuracy_score = correct_predictions / len(predicted_labels)

    # Return the accuracy score
    return accuracy_score * 100


def readfile(file_name):
    with open(file_name, "r") as file:
        return file.read()


def data_to_vectors(data, vocab):
    vectors = []
    labels = []

    for line in data.split("\n"):
        processed_line = process_text(line)

        if len(processed_line) > 0:
            (vector, label) = vectorize_text(processed_line, vocab)
            vectors.append(vector)
            labels.append(label)

    return (vectors, labels)


def createprocessedfile(vectors, labels, vocab, file_name):
    with open(file_name, "w") as file:
        # Write top line
        for item in vocab:
            file.write("%s," % item)

        file.write("\n")

        # Write each vector to file
        for i, vector in enumerate(vectors):
            for item in vector:
                file.write("%s," % item)

            file.write("%s\n" % labels[i])


def main():
    # Take in text files and outputs sentiment scores
    test_data = readfile("testSet.txt")
    training_data = readfile("trainingSet.txt")

    training_vocab = build_vocab(process_text(training_data))
    test_vocab = build_vocab(process_text(test_data))

    (training_vectors, training_labels) = data_to_vectors(training_data, training_vocab)
    (test_vectors, test_labels) = data_to_vectors(test_data, test_vocab)

    createprocessedfile(test_vectors, test_labels, test_vocab, "preprocessed_test.txt")
    createprocessedfile(
        training_vectors, training_labels, training_vocab, "preprocessed_train.txt"
    )

    classifier = BayesClassifier()

    # classifier.train(training_vectors, training_labels, training_vocab)
    # predictions = classifier.classify_text(training_vectors, training_vocab)

    # print("Accuracy for 100% of the training data: ", accuracy(predictions, training_labels))

    equal_parts = 4
    line_increments = len(training_data.split("\n")) / equal_parts

    with open("results.txt", "w") as file:
        file.write("The Results:")

    for i in range(equal_parts):
        if i == 0:
            sub_training_data = readfile("trainingSet1_1.txt")
        elif i == 1:
            sub_training_data = readfile("trainingSet1_2.txt")
        elif i == 2:
            sub_training_data = readfile("trainingSet1_3.txt")
        elif i == 3:
            sub_training_data = readfile("trainingSet1_4.txt")

        sub_training_vocab = build_vocab(process_text(sub_training_data))

        (sub_training_vectors, sub_training_labels) = data_to_vectors(
            sub_training_data, sub_training_vocab
        )

        createprocessedfile(
            sub_training_vectors,
            sub_training_labels,
            sub_training_vocab,
            "preprocessed_train.txt",
        )

        classifier.train(sub_training_vectors, sub_training_labels, sub_training_vocab)
        predictions = classifier.classify_text(training_vectors, training_vocab)

        accuracy_result = (
            "Accuracy for "
            + str((i + 1) * 25)
            + "% of the training data: "
            + str(accuracy(predictions, training_labels))
        )

        with open("results.txt", "a") as file:
            file.write("\n")
            file.write("\n")
            file.write(accuracy_result)
            file.write("\n")

        predictions = classifier.classify_text(test_vectors, training_vocab)
        accuracy_result = str(
            "Accuracy for the test data: " + str(accuracy(predictions, test_labels))
        )
        with open("results.txt", "a") as file:
            file.write(accuracy_result)

    return 1


if __name__ == "__main__":
    main()
