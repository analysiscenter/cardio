import sklearn


class LabelBinarizer(sklearn.preprocessing.LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if len(self.classes_) == 1:
            Y = 1 - Y
        if len(self.classes_) == 2:
            Y = np.hstack((1 - Y, Y))
        return Y

    def inverse_transform(self, Y, threshold=None):
        if len(self.classes_) == 1:
            y = super().inverse_transform(1 - Y, threshold)
        elif len(self.classes_) == 2:
            y = super().inverse_transform(Y[:, 1], threshold)
        else:
            y = super().inverse_transform(Y, threshold)
        return y
