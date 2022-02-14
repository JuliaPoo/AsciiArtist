"""Script to generate the model to choose the ascii chars

It uses the font specified in `./fonts` to generate images
of characters (distorted a little) to use as training data

The model is then saved at `asciiartist/gen/ascii-model.tflite`.
The onehot mapping is saved at `asciiartist/gen/_onehot.txt`.

The model is saved as .tflite so that `tflite-runner` can be used,
which minimises dependencies.

These files are packaged together for distribution. This
file however, isn't.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pathlib import Path


class DataGen:

    """
    Class that generates the training data.
    """

    def __init__(self, fz: int, font_path: str, chars: str, pad: int, batch_size: int, ratio: int = 2.5):

        self.fz = fz
        self.font = ImageFont.truetype(font_path, self.fz)
        self.chars = chars
        self.pad = pad
        self.ratio = ratio

        self._base_data = self._gen_base_data()
        self.h, self.w = self._base_data.shape[1:-1]

        x = int(self.fz*.1)
        self.augmentconfig = ImageDataGenerator(
            width_shift_range=(-x, x),
            height_shift_range=(-x, x),
            rotation_range=5,
            zoom_range=(0.9, 1.3),
            #shear_range = 0.5,
            fill_mode="constant", cval=0
        )
        self.augmentgen = self.augmentconfig.flow(
            self._base_data, [*self.chars],
            batch_size=batch_size,
            seed=0xfade1eaf
        )

        self.onehot = LabelBinarizer().fit([*self.chars])

    def _char2img(self, c: str):

        X, Y = int((self.fz+self.pad)/self.ratio+.5), self.fz+self.pad
        #X,Y = self.fz+self.pad, int((self.fz+self.pad)*self.ratio + .5)
        img = Image.new("RGB", (X, Y), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(c, font=self.font)
        draw.text(((X-w)//2, (Y-h)//2), c, (255, 255, 255), font=self.font)
        return img

    def _gen_base_data(self):
        return np.array([
            np.array(self._char2img(c), dtype=np.float32)
            for c in self.chars
        ])

    def batch(self):

        while True:
            X, y = self.augmentgen.next()
            X = X[:, :, :, 0:1]  # one channel
            X /= 256  # Shitty normalize
            yield X, self.onehot.transform(y)


class ASCIIModel():

    """
    Class that implements the model.

    Doesn't inherit from `Sequential` bcuz it slightly
    complicates the saving part.
    """

    def __init__(self, dg: DataGen):

        self.k5 = Sequential()

        self.dg = dg
        self.gen = dg.batch()

        self.k5.add(Conv2D(64, kernel_size=3, activation='relu',
                           input_shape=(dg.h, dg.w, 1)))
        self.k5.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.k5.add(Flatten())
        self.k5.add(Dense(len(dg.chars), activation='softmax'))

        self.k5.compile(optimizer='adam',
                        loss='categorical_crossentropy', metrics=['accuracy'])
        self.k5.fit(self.gen, steps_per_epoch=1 << 7, epochs=6)

    def print_report(self):

        Xt, yt = next(self.gen)
        yp = self.k5.predict(Xt)

        print(classification_report(
            self.dg.onehot.inverse_transform(yt),
            self.dg.onehot.inverse_transform(yp),
        ))

    def save(self, out_folder: str):

        # Write onehot
        open(Path(out_folder) / "_onehot.txt", "w") \
            .write("".join(self.dg.onehot.classes_))

        # Write model
        from tensorflow import lite
        converter = lite.TFLiteConverter.from_keras_model(self.k5)
        tfmodel = converter.convert()
        open(Path(out_folder) / "ascii-model.tflite", "wb") \
            .write(tfmodel)


def main():

    import os
    os.path.dirname(__file__)

    FONT_SIZE = 35
    FONT_PATH = r"fonts\consola.ttf"
    CHARS = "74ovcOYCTUVHFEKIL!\"#\'()*+,-./:;<=>@[\\]^_`{|}~ "
    PAD = 0

    dg = DataGen(FONT_SIZE, FONT_PATH, CHARS, PAD, 64)
    model = ASCIIModel(dg)
    model.print_report()
    model.save("../asciiartist/gen/")


if __name__ == "__main__":
    main()
