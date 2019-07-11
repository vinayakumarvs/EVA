# Receptive Field Calculation for GoogLeNet Network

| Sno  | Layer-Name   | Kernel | Padding | Stride | Input | Output   | Receptive Field  | Jump      |
| ---- | ------------ | ----- | ------ | ------- | ------ | ------ | ------- | ------- |
| 1    | Conv1        | 7   | 3      | 2       | 244      | **112**  | **7** | **1**   |
| 2    | MP1          | 3   | 0      | 2       | 112      | **56**  | **11**  | **2**  |
| 3    | Conv2        | 3    | 1      | 1       | 56      | **56**  | **19**  | **4**  |
| 4    | MP2          | 3    | 0      | 2       | 56      | **28**  | **27**  | **4**  |
| 5    | 3A_Inception | 5    | 2      | 1       | 28      | **28**  | **59**  | **8**  |
| 6    | 3B_Inception | 5    | 2      | 1       | 28      | **28**  | **91**  | **8**  |
| 7    | MP3          | 3    | 0      | 2       | 28      | **14**  | **107**  | **8** |
| 8    | 4A_Inception | 5    | 2      | 1       | 14      | **14** | **171**  | **16** |
| 9    | 4B_Inception | 5    | 2      | 1       | 14      | **14** | **235**  | **16** |
| 10   | 4C_Inception | 5    | 2      | 1       | 14      | **14** | **299**  | **16** |
| 11   | 4D_Inception | 5    | 2      | 1       | 14      | **14** | **363**  | **16** |
| 12   | 4E_Inception | 5    | 2      | 1       | 14      | **14** | **427**  | **16** |
| 13   | MP4          | 3    | 0      | 2       | 14      | **7** | **459**   | **16** |
| 14   | 5A_Inception | 5     | 2      | 1       | 7      | **7** | **587**   | **32** |
| 15   | 5B_Inception | 5     | 2      | 1       | 7      | **7** | **715**   | **32** |
| 16   | AvgPool      | 7     | 0      | 1       | 7      | **1** | **907**   | **32** |





#Have given an whirl to implement the Below given ENAS Network in an Enclosed Assignment_7.ipynb File.

