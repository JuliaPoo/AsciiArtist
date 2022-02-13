"""asciiartist module: Converts images to ascii art

Brief Usage Description:

This module's main export is `asciiartist`. This generates
ascii art from an image. `asciiartist` also enabled certain
configuration the user might wanna tune.

`asciiartist` returns the ascii art and a numpy array
representing the edges detected. `display_edges` is used
to display the edges, and this helps with debugging and
finetuning the parameters given to `asciiartist`.
"""

from typing import Tuple, Union

from skimage import feature
from PIL import Image
import numpy as np

def _model_predict(X:"np.ndarray[np.bool]", _model=[]
    ) -> "np.ndarray[np.uint8]":

    """Performs model inference
    
    Parameters
    ----------
    X: np.ndarray[np.bool]
        Input array, batched, representing edges for each 
        ascii char. This function will deal with batching.

    _model:
        Caches the loaded model, onehot mapping and in-out
        info.

    Returns
    -------
    np.ndarray[np.uint8]
        Char array of ascii characters.
    """

    batch_size = 32

    if len(_model) == 0:

        import os
        dir = os.path.dirname(__file__)

        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(
            model_path=os.path.join(dir, "gen/ascii-model.tflite"))
        m_out = interpreter.get_output_details()[0]
        m_in = interpreter.get_input_details()[0]
        
        interpreter.resize_tensor_input(
            m_in['index'], (batch_size, *m_in["shape"][1:]))
        interpreter.resize_tensor_input(
            m_out['index'], (batch_size, *m_out["shape"][1:]))
        interpreter.allocate_tensors()
        _model.append((
            interpreter,
            [*open(os.path.join(dir, "gen/_onehot.txt"), "rb").read()],
            (m_out, m_in)
        ))

    model, onehot, (m_out, m_in) = _model[0]

    asc = []
    for i in range(X.shape[0] // batch_size + 1):

        if (i+1)*batch_size > X.shape[0]:

            # Pad to `batch_size` samples
            pad = batch_size-X.shape[0]%batch_size
            model.set_tensor(
                m_in['index'],
                np.pad(
                    X[i*batch_size: (i+1)*batch_size], 
                    [(0,pad), (0,0), (0,0), (0,0)], 
                    mode="constant"
                ))

            model.invoke()
            y = model.get_tensor(m_out['index'])[:-pad]
            asc.append(np.argmax(y, axis=1))
            continue

        model.set_tensor(
            m_in['index'], X[i*batch_size: (i+1)*batch_size])

        model.invoke()
        y = model.get_tensor(m_out['index'])
        asc.append(np.argmax(y, axis=1))
        
    return np.array([onehot[i] for i in np.concatenate(asc)], dtype=np.uint8)


def _get_edges(
    img: 'Image', 
    height: int, 
    sigma: float, 
    weight: float, 
    ratio: float
    ) -> "np.ndarray[np.bool]":

    """Detects edges and segments image.
    
    Parameters
    ----------
    img: 'Image'
        Image object from Pillow (PIL.Image)

    height: int
        Number of lines of output art

    sigma: float
        Noise reduction param for canny edge detection

    weight: float
        Line weight. A higher value thickens the edges

    ratio: float
        y/x ratio of each char of text

    Returns
    -------
    np.ndarray[np.uint8]
        5D array representing edges detected per segment
        Shape: (y,x,ph,pw,1)
        y,x ==> Height and width of image in ascii chars
        ph,pw ==> Height and width of each ascii char in 
            image pixels.
    """

    img = np.array(img.convert("L"))
    y, x = img.shape[:2]
    width = int(x/y*height*ratio+.5)
    
    ph,pw = 88,35

    # Resize image
    ny = ph*height
    nx = pw*width
    img = np.array(
        Image.fromarray((img.astype(float)*255).astype(np.uint8))
        .resize((int(nx//weight+.5), int(ny//weight+.5)), Image.NEAREST),
        dtype=np.uint8)

    edges = feature.canny(img, sigma=sigma) \
        [3:-3,3:-3] # Remove border (scikit-image bug)
    edges = np.array(
        Image.fromarray(edges)
        .resize((nx, ny), Image.NEAREST)
    ).astype(np.float32)
    edges = edges.reshape(*edges.shape, 1)

    segs = np.array([
        [edges[ph*i:ph*(i+1), pw*j:pw*(j+1)] for j in range(width)]
        for i in range(height)
    ])

    return segs


def _segments_to_ascii(edges: "np.ndarray[np.bool]") -> str:

    """Formats input and output of model

    Parameters
    ----------
    edges: np.ndarray[np.bool]
        Output of `_get_edges`. See docs of `_get_edges`.

    Returns
    -------
    str
        ascii art string
    """

    y, x = edges.shape[:2]
    inp = edges.reshape(-1, *edges.shape[-3:])
    asc = _model_predict(inp)
    return "\n".join("".join(map(chr, i)) for i in asc.reshape(y, x))


def display_edges(
    edges: "np.ndarray[np.bool]"
    ) -> None:

    """Displays edges (2nd output of `asciiartist`)
    
    This can be used to manually finetune `asciiartist` 
    parameters.

    The edges will be displayed, and grid segments the images
    for where each ascii char will be placed.

    Parameters
    ----------
    edges: np.ndarray[np.bool]
        2nd output of `asciiartist`. See `asciiartist` docs
    """

    import matplotlib.pyplot as plt

    y,x,h,w,_ = edges.shape
    eimg = np.hstack(np.hstack(edges))
    eimg = eimg.reshape(eimg.shape[:-1])

    plt.figure(figsize=(10,10))
    plt.imshow(eimg.astype(float))
    for i in range(y):
        plt.plot([0,x*w],[i*h,i*h], color="white", linewidth=0.1)
    for i in range(x):
        plt.plot([i*w,i*w],[0,y*h], color="white", linewidth=0.1)
        
    plt.ylim((y*h,0))
    plt.xlim((0,x*w))
    plt.show()


def asciiartist(
    img: 'Image', 
    n_lines: int, 
    noise_reduction: float = 3, 
    line_weight: float = 1, 
    text_ratio: float = 2.5,
    _generate_ascii: bool = True
    ) -> "Tuple[Union[str, None], np.ndarray[np.bool]]":

    """Converts images into ascii art
    
    Returns two items (ascii_art, edges). `edges` can be
    displayed with `display_edges` to aid in manually 
    finetuning the parameters.

    Parameters
    ----------
    img: 'Image'
        Image object from Pillow (PIL.Image)

    n_lines: int
        Number of lines of output art

    noise_reduction: float, optional
        Noise reduction param for canny edge detection
        (Default 3.)

    line_weight: float, optional
        Line weight. A higher value thickens the edges
        (Default 1.)

    text_ratio: float, optional
        y/x ratio of each char of text. (Default 2.5)

    _generate_ascii: bool, optional
        If disabled, no ascii art is generated, but function
        will return the edges. This option can be used if 
        the user only wants the `edges` output to put into
        `display_edges` for manual finetuning of the 
        parameters (Default True)

    Returns
    -------
    (str | None, np.ndarray[np.bool])

        Let the output be (art, edges).

        `art` is the ascii art string. This value is None
        if `_generate_ascii` is false.

        `edges` is a numpy array representing the edges
        detected. This can be fed into `display_edges` to aid
        in manually finetuning the parameters.

    """
    
    edges = _get_edges(
        img, n_lines,
        sigma = noise_reduction,
        weight = line_weight,
        ratio = text_ratio
    )

    asc = None
    if _generate_ascii:
        asc = _segments_to_ascii(edges)

    return asc, edges