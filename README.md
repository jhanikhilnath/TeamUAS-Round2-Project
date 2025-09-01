# Team UAS Round 2 Project

## Usage

1. Download the entire code base from GitHub

2. Install the required dependencies using
   `pip install -r requirements.txt`

3. Ensure that all the input images are in a folder named `uas images`

4. Run the code using
   `python main.py`

## Output

1. A folder `output_folder/id_marked` will contain all the IDs assigned to the various Rescue Centers(C1, C2,...) as well as the various patients(P1, P2,...) will display on the screen

2. Another Folder `output_folder/segmented_colors` will contain image showing segmented parts of land and water will be outputed

3. Text output containing all distance matrix, Center Score List as well as the cummalative score will be shown for each and every image orderwise

4. The final list of all the center scores assigned will also be displayed groupwise in the order `[blue, pink, grey]`

5. A sorted list of all the images will be shown at the end (with the first image having the highest possible score)

## Algorithm Used

Calculation of the score was done using the formula `Priority Order/Distance`
Where priority order is nothing but `Age  Severity * Emergency order`.

Age Severity has been assigned on the basis of [Star, Triangle, Square] where star has the highest priority and square has the lowest.

Emergency Order has been assigned on the basis of [red, yellow, green] where red has the highet priority and green has the lowest.
