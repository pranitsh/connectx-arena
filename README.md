# connectx-arena

A lightweight web arena for playing Connect 4 against pre-trained Reinforcement Learning agents. The goal of this project is to provide a live, frictionless environment to test model heuristics, benchmark agent performance, and explore how different neural network architectures approach the game.

## The Opponents

The arena hosts five distinct architectures, all balanced to have roughly similar total parameter counts to test depth, width, and spatial processing:

* **mlp-small:** `[64, 64]`. The shallowest model, using just two layers of 64 neurons.
* **mlp-medium:** `[40, 40, 40, 40]`. A balanced, mid-sized model utilizing four layers of 40 neurons.
* **mlp-large:** `[32, 32, 32, 32, 32, 32]`. The deepest model, utilizing six layers of 32 neurons.
* **cnn-small:** 16 Conv2D filters -> 342 dense. A small-scale convolutional model using 16 spatial filters.
* **cnn-large:** 32 Conv2D filters -> 342 dense. A larger convolutional model using 32 spatial filters.

## Architecture & Security

This project runs a serverless stack designed to protect compute-heavy ML inference without requiring users to log in:

* **Frontend:** Vanilla HTML and JavaScript hosted purely on GitHub Pages.
* **Backend:** Python Firebase Cloud Functions running model inference.
* **Security:** Firebase Anonymous Auth issues silent session IDs, while Firebase App Check (reCAPTCHA v3) verifies real web traffic to prevent API abuse.

## Local Setup

To run the frontend locally and test the UI:

1. Install the base Firebase dependencies:
   ```bash
   npm install
   ```

2. Start the local server to serve the `public/` folder:
   ```bash
   npm run start
   ```
