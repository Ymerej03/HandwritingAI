import os

# for each image in the labelled_images folder need to add to a folder with subfolders
# jeremy_words/z01/z01-000/
# images named according to the folder then simply increasing
# ie z01-000-00-00.png
# then need to fill out the rest of the parameters according to the iam dataset method
# so to write to a text file called jeremy_words.txt
# ie z01-000-00-00 ok 99 99 99 99 99 aa "transcription"
# the transcription is obtained from the original name of the image split on the _

input_folder = "labelled_images"
output_folder = "jeremy_words_dataset/jeremy_words"
output_text_folder = output_folder.split("/")[0]

# Create the output folder if it doesn't exist
if not os.path.exists(output_text_folder):
    print("Specified output folder does not exist, creating it now...")
    os.makedirs(output_text_folder)

if not os.path.exists(output_folder):
    print("Specified output folder does not exist, creating it now...")
    os.makedirs(output_folder)

# Iterate through images in the input directory
for i, image_name in enumerate(os.listdir(input_folder)):
    # Create output file path
    output_number = f"{i:07d}"
    a = output_number[:3]
    b = output_number[3:5]
    c = output_number[5:]
    top_folder = "x03"
    new_image_name = f"{top_folder}{os.sep}{top_folder}-{a}{os.sep}{top_folder}-{a}-{b}-{c}.png"
    output_file_path = os.path.join(output_folder, new_image_name)

    # Create subdirectories if they don't exist
    os.makedirs(os.path.join(output_folder, f"{top_folder}{os.sep}{top_folder}-{a}"), exist_ok=True)

    # Write to text file
    with open(f"{output_text_folder}/jeremy_words.txt", "a") as f:
        transcription = image_name.split('_')[0]
        f.write(f"{new_image_name.split('.')[0].split(os.sep)[-1]} ok 99 99 99 99 99 aa {transcription}\n")

    # Copy the image to the output directory
    os.rename(os.path.join(input_folder, image_name), output_file_path)

print("Processing complete.")
