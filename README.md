# Evo painter

## Application
### Evo painter - it's a window app that uses evolutionary algorithms to recreate and stylize your images. Upload a picture, run the evolution, and watch the algorithm generate a copy of it in real time using mosaics, fractals, or text.
    
### 1 - Features
- 🎨 Upload images: PNG, JPG, JPEG, BMP.
- ⚙️ Smart pre-processing: Automatically crop or resize your image to fit into a square format.
- 🧬 Multiple algorithms: Various genetic algorithms have been implemented to generate:
  - Mosaic: Create an image from colored blocks.

  - Text: (Coming later)

  - Fractals: (Coming later)
- 👀 Real-time visualization: Update the best result every 100ms.
- ⏯️ Process control: Start, pause the algorithm at any time.
- 📊 Similarity score: After stopping, the fitness function value (similarity metric to the original) is displayed.
- 💾 Saving results: Save the generated image in the desired location and format.

### 2 - Installation
- Prerequisites
    - Python 3.12 or later

    - pip (Python package manager)

### 3 - How it works?
1. Target Loading: The user uploads a target image.

2. Preparation: The algorithm scales and processes the image.

3. Population Initialization: An initial population of random "individuals" is created (e.g. sets of colored blocks for a mosaic).

4. Evolution Cycle:

    - Evaluation: Each individual is evaluated by a fitness function (e.g. MAE - mean absolute error) that measures its similarity to the target image.

    - Selection: The best individuals (elite) are selected to create the next generation.

    - Crossover: Parent individuals "cross" by exchanging their genes (parts of the image).

    - Mutation: Random changes are made to the new individuals (change pixel color, swap blocks, etc.).

5. Repetition: Steps 4a-4d are repeated hundreds and thousands of times, gradually reducing the error and improving the result.

### 4 - Dependencies
- PyQt6 — for the GUI.

- Pillow (PIL) — for working with images.

- numpy — for important math calculations with arrays.

- scikit-image — for advanced image comparison metrics (e.g. rgb2lab).

- pygame — (only uses cursors, may want to reconsider).


### 5 - License
- This project is licensed under the MIT License. See the LICENSE file for more information.





