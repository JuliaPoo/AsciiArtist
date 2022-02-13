# Ascii Artist

An ascii art generator that's actually good. 
Does edge detection and selects the most appropriate characters.


<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaPoo/AsciiArtist/main/rsrc/skykid.jpg" alt = "Blue Tit">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaPoo/AsciiArtist/main//rsrc/bluetit.jpg" alt = "Blue Tit">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaPoo/AsciiArtist/main//rsrc/niko.jpg" alt = "Niko from oneshot!">
</p>

## Installing

### Installing with pip:

```
pip install asciiartist
```

### Installing from wheel:

Download the wheel file from this project's releases and run 

```
pip install <path/to/wheel>
```

## Quick Start

```py
from asciiartist import asciiartist, display_edges
from PIL import Image

img = Image.open("niko.png")

art, edges = asciiartist(
    img, # The image!
    30,  # Number of lines of the output ascii art
    noise_reduction=3,  # Level of noise reduction (optional)
    line_weight=1,      # Weight of the lines to draw (optional)
    text_ratio=2.2      # Height/width ratio of each character (optional)
)

print(art) # `art` is a string u can just print

# v Display the result of edge detection. 
#   Good for finetuning params.
display_edges(edges)
```

## Build from Source

Run the script `./model/model-gen.py` and build the wheel with `poetry build -f wheel`.

## How it works

Roughly, how _Ascii Artist_ generates the drawings:

1. Run edge detection
2. Segment the image for each char
3. Pass each segment through a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) to get the most appropriate character.

The CNN is trained with distorted images of characters (in font consolas),
created in a way that emulates the output of the edge detection.

## Bye

^-^

```
                                          __=E.L__      
                                        >#-=@>@F* `<_   
                                      _/`@o<FTFC@[_~_.__
                                   _./`\_`<__      @@7*`
                              _,~``      *v `^L_  )(    
                         _,~^C___ _    ````*@```````    
                   __,;@"*'`C` *@@_       _-V           
               __,@'^`CC)-[_L-----o,,}<@F--             
    __,-Ec>C<4@'C`'`"-[_,/   _,L-L             ,"       
 -`````          ````                       _-'         
                        _   _           __~``           
                        ``  "<`  _                      
                              `(V\_                     
                                `V(<_                   
                   _____,,~~<7oEE(@@_Eo@@Fo,            
       ___,-~-^````               .-.__V)  ,_           
```