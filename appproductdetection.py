import streamlit as st
from fastai.vision.all import load_learner, PILImage
import recipe_generator
import videoprediction
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from pathlib import Path
from PIL import Image
import warnings
from geminiapi import detect_product_names
from yoloimages import  display_detected_frames , display_best_detected_image
warnings.filterwarnings("ignore", category=DeprecationWarning)
from duckduckgo import search_images , display_image_from_url , sanitize_filename

# Set the path where images will be saved
path = Path('foodproduct')
path.mkdir(exist_ok=True)



# Custom CNN model definition
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

# Load the model
model_filename = 'models/fruit_cnn_model.pkl'
if not Path(model_filename).exists():
    st.error(f"Model file '{model_filename}' not found.")
    learn=""
else:
    learn = load_learner(model_filename)

# Function for ingredient detection from image
# Function for ingredient detection from image
def detect_ingredients_from_image(uploaded_image):
    img = PILImage.create(uploaded_image)
    product, _, probs = learn.predict(img)
    return product, probs

# Function to list products above a probability threshold
def list_all_products(probs, threshold=0.7):
    return [(learn.dls.vocab[i], probs[i].item()) for i in range(len(probs)) if probs[i] > threshold]

def predict_image(model, image_path, transform, class_names):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB
    image = transform(image).unsqueeze(0)  # Apply transformation and add batch dimension

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # This will call model.forward internally

    # Get the predicted class (index of the maximum logit)
    _, predicted_class_idx = torch.max(output, 1)

    # Return the predicted class name
    return class_names[predicted_class_idx.item()]

# Define real class names
CLASS_NAMES = [
    'Blueberries', 'Broccoli', 'Pasta', 'Chicken Breast',
    'Fillet Salmon', 'Grounded Beef', 'Avocado', 'Banana',
    'Carrot', 'Mushrooms', 'Cucumber',
    'Garlic', 'Lemon', 'Orange', 'Pineapple',
    'Apple', 'Strawberries', 'Sweet Potato',
    'Tomatoe', 'Onion', 'Bell Pepper',
    'Potato', 'Lettuce', 'Cheese', 'Eggs'
]

# Load the pre-trained transfer modeling model
class ImagenetTransferModeling(nn.Module):
    def __init__(self, num_classes=25):
        super(ImagenetTransferModeling, self).__init__()
        # Pre-trained ResNet-50
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]  # Remove the fully connected layer
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x

# Load the trained model from a .pth file
model_path = "models/model_and_transform.pth"
if Path(model_path):
    model2 = ImagenetTransferModeling(num_classes=25)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()

    # Load the transform
    transform_params = checkpoint.get("transform_params", {"resize": (224, 224)})
    transform = transforms.Compose([
        transforms.Resize(transform_params["resize"]),
        transforms.ToTensor(),
    ])
else:
    model2 = None
    st.error("Model file not found. Please check the file path.")

# Prediction function for fridge sections
def classify_image_in_parts(image_path, transform, num_shelves):
    model2.eval()  # Ensure model is in evaluation mode
    image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
    width, height = image.size
    section_height = height // num_shelves

    sections_predictions = []
    for i in range(num_shelves):
        top = i * section_height
        bottom = top + section_height if i < num_shelves - 1 else height
        section = image.crop((0, top, width, bottom))
        transformed_section = transform(section).unsqueeze(0)

        with torch.no_grad():
            logits = model2(transformed_section)  # This calls model.forward automatically
            probabilities = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=num_items, dim=1)

        if num_items == 1:
            top_probs = [top_probs.item()]  # Wrap single value in a list
            top_indices = [top_indices.item()]  # Wrap single value in a list
        else:
            top_probs = top_probs.squeeze().tolist()
            top_indices = top_indices.squeeze().tolist()

        section_predictions = [(idx, prob) for idx, prob in zip(top_indices, top_probs)]
        sections_predictions.append({"section": i + 1, "predictions": section_predictions})

    return sections_predictions

# Streamlit UI
st.title("🍽️ Recipe Generator with Fridge Section Analysis")

st.sidebar.header("Choose Detection Method")
option = st.sidebar.selectbox("Select a method:", ["Image Detection", "Image Detection with API", "Video Detection"])

if option == "Image Detection":
    st.header("📸 Image Upload for Fridge Analysis")
    uploaded_image = st.file_uploader("Upload a fridge image:", type=["jpg", "jpeg", "png"])

    # Initialize session state for detected products and ingredients list if not already set
    if "detected_products" not in st.session_state:
        st.session_state.detected_products = set()

    if "ingredients_list" not in st.session_state:
        st.session_state.ingredients_list = []

    # Only proceed if the user has uploaded an image
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Number of shelves input
        num_shelves = st.number_input("How many shelves are in the fridge?", min_value=1, value=3)
        num_items = st.number_input("How many items on each shelf?", min_value=1, value=2)

        # Detect Ingredients Button (Detect products from the uploaded image)
        if st.button("🔍 Detect Ingredients"):
            predictions = classify_image_in_parts(uploaded_image, transform, num_shelves)

            # Gather detected ingredients, omitting duplicates
            detected_products = set(
                CLASS_NAMES[idx] for pred in predictions for idx, _ in pred["predictions"]
            )
            st.session_state.detected_products = detected_products

            # Display predictions by shelf
            for prediction in predictions:
                st.subheader(f"Shelf {prediction['section']} Predictions")
                for idx, prob in prediction["predictions"]:
                    st.write(f"{CLASS_NAMES[idx]}: {100*prob:.2f}%")

            # If products are detected, allow the user to edit and confirm
            if st.session_state.detected_products:
                detected_products_text = ", ".join(sorted(st.session_state.detected_products))
                st.text_area("Detected Ingredients:", detected_products_text, height=100, disabled=True)
                ingredients_input = st.text_area("✏️ Edit Detected Ingredients (optional):", detected_products_text)
                if ingredients_input != detected_products_text:
                    st.session_state.ingredients_list = [ingredient.strip() for ingredient in ingredients_input.split(",") if ingredient.strip()]
                else:
                    st.session_state.ingredients_list = list(st.session_state.detected_products)

        # Only allow the user to generate a recipe after ingredients are confirmed
        if st.session_state.ingredients_list:
            if st.button("🔍 Generate Recipe"):
                try:
                    # Generate the recipe using the ingredients list
                    recipe = recipe_generator.generate_recipe(st.session_state.ingredients_list)

                    if recipe:
                        st.subheader("📜 Generated Recipe")
                        st.write(f"**Title:** {recipe.title} 🥘")

                        # Search and show recipe image after generating the recipe
                        search_term = recipe.title  # Use recipe title or a key ingredient for image search
                        image_urls = search_images(search_term, max_images=1)  # Get the first image URL

                        if image_urls:
                            # Show the image with a fixed size of 100x100 pixels
                            display_image_from_url(image_urls[0])
                        else:
                            st.write("No image found for the recipe.")

                        st.write(f"**Servings:** {recipe.servings} 🥣")
                        st.write(f"**Preparation Time:** {recipe.prep_time} ⏱️")
                        st.write(f"**Cooking Time:** {recipe.cook_time} ⏲️")
                        st.write(f"**Total Time:** {recipe.total_time} ⏳")
                        st.write(f"**Difficulty:** {recipe.difficulty} 🏆")
                        st.write(f"**Cuisine:** {recipe.cuisine} 🌍")
                        st.write(f"**Category:** {recipe.category} 📂")

                        st.subheader("🛒 Ingredients")
                        for ing in recipe.ingredients:
                            st.write(f"- {ing.amount} {ing.unit} {ing.name} 🥄")

                        st.subheader("🔪 Instructions")
                        for index, instruction in enumerate(recipe.instructions, start=1):
                            st.write(f"{index}. {instruction}")

                        # Added and removed products
                        st.subheader("🛍️ Added Products")
                        added_products_text = ", ".join(
                            recipe.added_products) if recipe.added_products else "No added products found."
                        st.text_area("Added Products", value=added_products_text, height=100)

                        st.subheader("❌ Removed Products")
                        removed_products_text = ", ".join(
                            recipe.removed_products) if recipe.removed_products else "No removed products found."
                        st.text_area("Removed Products", value=removed_products_text, height=100)


                    else:
                        st.error("No recipe found.")
                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")

elif option == "Image Detection with API":
    st.header("📸 Gemini - Image Upload for Fridge Analysis-")
    uploaded_image = st.file_uploader("Upload a fridge image:", type=["jpg", "jpeg", "png"])

    # Initialize session state for detected products and ingredients list if not already set
    if "detected_products" not in st.session_state:
        st.session_state.detected_products = set()

    if "ingredients_list" not in st.session_state:
        st.session_state.ingredients_list = []

    # Only proceed if the user has uploaded an image
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Detect Ingredients Button (Detect products from the uploaded image)
        if st.button("🔍 Detect Ingredients"):
            # Assuming detect_product_names returns a list of detected products

            temp_file = Path(f"temp_{uploaded_image.name}")
            with open(temp_file, "wb") as f:
                f.write(uploaded_image.getbuffer())

            detected_products = detect_product_names(str(temp_file))

            if detected_products:
                st.session_state.detected_products.update(
                    detected_products)  # Update session state with detected products
                detected_products_text = ", ".join(sorted(st.session_state.detected_products))
                st.text_area("Detected Ingredients:", detected_products_text, height=100, disabled=True)

                # Allow the user to edit the detected ingredients
                ingredients_input = st.text_area("✏️ Edit Detected Ingredients (optional):", detected_products_text)

                # Update the ingredients list based on user input
                if ingredients_input != detected_products_text:
                    st.session_state.ingredients_list = [ingredient.strip() for ingredient in
                                                         ingredients_input.split(",") if ingredient.strip()]
                else:
                    st.session_state.ingredients_list = list(st.session_state.detected_products)
            else:
                st.warning("No ingredients detected. Please try uploading a different image.")

        # Only allow the user to generate a recipe after ingredients are confirmed
        if st.session_state.ingredients_list:
            if st.button("🔍 Generate Recipe"):
                try:
                    # Generate the recipe using the ingredients list
                    recipe = recipe_generator.generate_recipe(st.session_state.ingredients_list)

                    if recipe:
                        st.subheader("📜 Generated Recipe")
                        st.write(f"**Title:** {recipe.title} 🥘")

                        # Search and show recipe image after generating the recipe
                        search_term = recipe.title  # Use recipe title or a key ingredient for image search
                        image_urls = search_images(search_term, max_images=1)  # Get the first image URL

                        if image_urls:
                            # Show the image with a fixed size of 100x100 pixels
                            display_image_from_url(image_urls[0])
                        else:
                            st.write("No image found for the recipe.")

                        st.write(f"**Servings:** {recipe.servings} 🥣")
                        st.write(f"**Preparation Time:** {recipe.prep_time} ⏱️")
                        st.write(f"**Cooking Time:** {recipe.cook_time} ⏲️")
                        st.write(f"**Total Time:** {recipe.total_time} ⏳")
                        st.write(f"**Difficulty:** {recipe.difficulty} 🏆")
                        st.write(f"**Cuisine:** {recipe.cuisine} 🌍")
                        st.write(f"**Category:** {recipe.category} 📂")

                        st.subheader("🛒 Ingredients")
                        for ing in recipe.ingredients:
                            st.write(f"- {ing.amount} {ing.unit} {ing.name} 🥄")

                        st.subheader("🔪 Instructions")
                        for index, instruction in enumerate(recipe.instructions, start=1):
                            st.write(f"{index}. {instruction}")

                        # Added and removed products
                        st.subheader("🛍️ Added Products")
                        added_products_text = ", ".join(
                            recipe.added_products) if recipe.added_products else "No added products found."
                        st.text_area("Added Products", value=added_products_text, height=100)

                        st.subheader("❌ Removed Products")
                        removed_products_text = ", ".join(
                            recipe.removed_products) if recipe.removed_products else "No removed products found."
                        st.text_area("Removed Products", value=removed_products_text, height=100)


                    else:
                        st.error("No recipe found.")
                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")


elif option == "Video Detection":
    # Video upload section
    st.header("🎥 Video Upload for Ingredient Detection")
    uploaded_video = st.file_uploader("Upload a video of ingredients:", type=["mp4", "mov"])

    if uploaded_video:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Show the uploaded video
        st.video("temp_video.mp4")
        st.success("✨ Video uploaded successfully!")

        # Allow the user to adjust parameters using Streamlit widgets
        st.sidebar.subheader("🛠️ Adjust Detection Settings")

        # Parameter Inputs
        frame_skip = st.sidebar.slider("Frame Skip (frames)", min_value=1, max_value=50, value=20, step=1)
        input_size = st.sidebar.slider("Input Size (pixels)", min_value=128, max_value=640, value=320, step=32)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        nms_threshold = st.sidebar.slider("NMS Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

        # Run predictions on the video with adjusted parameters
        video_predictions, detected_frames = videoprediction.process_video(
            "temp_video.mp4", learn, "output_video.mp4",
            frame_skip=frame_skip,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        if video_predictions:
            st.markdown("### 🎉 Ingredients Detected! ✨")
            st.markdown("Here are the ingredients detected in the video:")

            detected_ingredients = []
            for product, time_set in video_predictions.items():
                formatted_times = videoprediction.format_time_list(time_set)
                st.markdown(f"**🍽️ {product}** - ⏰ Seconds: {formatted_times}")
                detected_ingredients.append(product)

            st.markdown("---")  # Add a horizontal line for separation

            # Add radio button to choose between generating a recipe or showing image detection
            choice = st.radio(
                "Choose an action:",
                ("None", "Generate Recipe", "Show Full Image Detection", "Show Cropped Image Detection", "Show Highlighted Video Frames")
            )

            if choice == "Generate Recipe":
                # Generate recipe based on detected ingredients
                if detected_ingredients:
                    try:
                        recipe = recipe_generator.generate_recipe(detected_ingredients)

                        if recipe:
                            st.subheader("📜 Generated Recipe")
                            st.write(f"**Title:** {recipe.title} 🥘")
                            st.write(f"**Servings:** {recipe.servings} 🥣")
                            st.write(f"**Preparation Time:** {recipe.prep_time} ⏱️")
                            st.write(f"**Cooking Time:** {recipe.cook_time} ⏲️")
                            st.write(f"**Total Time:** {recipe.total_time} ⏳")
                            st.write(f"**Difficulty:** {recipe.difficulty} 🏆")
                            st.write(f"**Cuisine:** {recipe.cuisine} 🌍")
                            st.write(f"**Category:** {recipe.category} 📂")

                            st.subheader("🛒 Ingredients")
                            for ing in recipe.ingredients:
                                st.write(f"- {ing.amount} {ing.unit} {ing.name} 🥄")

                            st.subheader("🔪 Instructions")
                            for index, instruction in enumerate(recipe.instructions, start=1):
                                st.write(f"{index}. {instruction}")

                    except Exception as e:
                        st.error(f"❗ An error occurred: {str(e)}")
                else:
                    st.warning("⚠️ No ingredients detected to generate a recipe.")

            elif choice == "Show Full Image Detection":
                # Detect products in the video and display the detected frames
                cropped_detections, detected_images = videoprediction.detect_all_products("temp_video.mp4", learn)
                display_detected_frames(detected_images)
            elif choice == "Show Cropped Image Detection":
                # Detect products in the video and display the detected frames
                cropped_detections, detected_images = videoprediction.detect_all_products("temp_video.mp4", learn)
                display_best_detected_image(cropped_detections)
            elif choice ==  "Show Highlighted Video Frames":
                # Detect and highlight elements in video frames
                highlighted_video = videoprediction.detect_and_highlight_elements(
     "temp_video.mp4", learn,
                frame_skip=5,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold   )

        else:
            st.warning("⚠️ No ingredients detected in the video. Please try another video.")
            st.markdown("🔍 **Tip:** Ensure the video is clear and the ingredients are visible!")
