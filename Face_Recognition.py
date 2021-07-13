import face_recognition
from PIL import Image, ImageDraw

image_of_ben = face_recognition.load_image_file('./img/known/Ben Affleck.jpg')
ben_face_encoding = face_recognition.face_encodings(image_of_ben)[0]

image_of_henry = face_recognition.load_image_file('./img/known/Henry Cavill.jpg')
henry_face_encoding = face_recognition.face_encodings(image_of_henry)[0]

image_of_scarlett = face_recognition.load_image_file('./img/known/Scarlett Johansson.jpg')
scarlett_face_encoding = face_recognition.face_encodings(image_of_scarlett)[0]

#  Create arrays of encodings and names
known_face_encodings = [
  ben_face_encoding,
  henry_face_encoding,
  scarlett_face_encoding
]

known_face_names = [
  "Ben Affleck",
  "Henry Cavill",
  "Scarlett Johansson"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('./img/groups/henry-cavill-ben-affleck.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('Result.jpg')