# Lab Materia Condensata

## Overview
This repository contains materials and code for the Lab Materia Condensata course. The course focuses on condensed matter physics and includes various experiments and simulations.

## Table of Contents
- [Lab Materia Condensata](#lab-materia-condensata)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Experiments](#experiments)
  - [Todo](#todo)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction
The Lab Materia Condensata course aims to provide hands-on experience with condensed matter physics. This repository includes all necessary resources, such as experiment protocols, data analysis scripts, and simulation codes.
The script `cleaner_script.py` is a useful script to clean the data. After a measurement call 
```bash
python cleaner_script.py path/to/file
```
to clean the files located at `path/to/file` and save it (automatically) in `Clean_Data`. One needs to manually put them inside a folder.

## Experiments
Detailed descriptions and protocols for each experiment can be found in the `Raw_Data` directory.
The file name of each measure specifies some information of such measure, as an example `BB_RAD_300ms_700V_10p5tac_303Hz__5mil6mil_241024` means:
| Field       | Description                        |
|-------------|------------------------------------|
| BB          | Experiment type                    |
| RAD         | Measurement type                   |
| 300ms       | Lockin integration time            |
| 700V        | Voltage across the phototube       |
| 10p5tac     | Opening of the slits               |
| 303Hz       | Lock in frequency                  |
| 5mil6mil    | Wavelenght range                   |
| 241024      | Unique identifier or timestamp     |

## Todo
- [ ] The names of the measurements in the Field / Description model are inconsistent across different measurements.
- [ ] Changed the folder structure, everything is broken for the moment. Fix it.
  - [x] Fix the notebook.
  - [ ] Fix `modules.calibration` and `modules.responce`.
- [ ] Add a python script that cleans the data and adds it in the correct folder.

## Installation
To set up the environment for running the simulations and analysis scripts, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/zar-hub/Lab_Materia_Condensata.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd Lab_Materia_Condensata
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Instructions on how to run the experiments and simulations can be found in their respective directories. Refer to the README files within each folder for detailed usage guidelines.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.