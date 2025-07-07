Here is a detailed to-do list, broken down by each section of the project analysis.

1. Data In

- [ ] Screenshot OCR Pipeline:
  - [ ] Create a new class MLBBScreenReader.
  - [ ] Implement template matching or a simple CV model to find and crop the main scoreboard region from the screenshot.
  - [ ] Within that scoreboard crop, hard-code the relative coordinates for:
    - [ ] KDA
    - [ ] GPM
    - [ ] Hero Damage
    - [ ] Match Duration
  - [ ] Feed these smaller, specific crops to the EasyOCR engine.
  - [ ] Replace the existing from_screenshot logic with this new, more robust pipeline.
- [ ] Replay/API Strategy (Decision):
  - [ ] Decide: Choose one primary method for advanced data collection (replay binary reverse-engineering vs. screen recording analysis).
  - [ ] Prototype: Begin initial work on the chosen method.

2. Engine Logic

- [ ] Threshold Optimizer:
  - [ ] Add a mechanism to the API to receive user feedback (e.g., a POST /feedback endpoint).
  - [ ] Create a ThresholdOptimizer class.
  - [ ] Implement logic to adjust values in thresholds.yml based on received feedback (e.g., using an exponential moving average).
- [ ] Hero Rule Autogenerator:
  - [ ] Create a new standalone Python script (e.g., scripts/scaffold_hero.py).
  - [ ] The script should take a hero name and role as command-line arguments.
  - [ ] It should copy the appropriate role template to create a new hero rule file.
  - [ ] It should add the new hero to the thresholds.yml file with default role values.
- [ ] Mental Coach Enhancement:
  - [ ] Modify MentalCoach.get_mental_boost to accept the statistical feedback as an argument.
  - [ ] Add logic to identify the most critical "warning" or "critical" feedback.
  - [ ] Append a specific, actionable suggestion to the mental boost message that relates to that critical feedback.

3. Delivery Layer

- [ ] FastAPI Wrapper:
  - [ ] Flesh out web/app.py.
  - [ ] Create a POST /analyze endpoint.
  - [ ] The endpoint should accept match data (initially as JSON, later from a file upload).
  - [ ] It should call the existing generate_feedback function.
  - [ ] It should return the feedback as a JSON response, including the severity levels.
- [ ] React/Tailwind UI:
  - [ ] Set up a new frontend project using Vite (npm create vite@latest).
  - [ ] Install and configure Tailwind CSS.
  - [ ] Create a simple file upload component for the screenshot.
  - [ ] Create a results page that can render the JSON feedback from the API, using different styles for different severity levels.
- [ ] Dockerization:
  - [ ] Create a Dockerfile for the Python backend.
  - [ ] Ensure all dependencies from requirements.txt are installed correctly.
  - [ ] Create a docker-compose.yml file.
  - [ ] Configure it to build and run the FastAPI application.
  - [ ] (Optional) Add a service for an Nginx reverse proxy.

4. Quality + Ops

- [ ] Structured Logging:
  - [ ] Add loguru to requirements.txt.
  - [ ] Replace all print() and logging.basicConfig() calls with loguru's logger for consistent, structured output.
- [ ] Strict Type Checking:
  - [ ] Add mypy to a requirements-dev.txt file.
  - [ ] Create a mypy.ini configuration file.
  - [ ] Run mypy --strict . and fix all reported type errors.
- [ ] CI Pipeline (GitHub Actions):
  - [ ] Create a new workflow file in .github/workflows/.
  - [ ] Add steps to:
    - [ ] Check out the code.
    - [ ] Install Python dependencies.
    - [ ] Run the linter (ruff or flake8).
    - [ ] Run mypy --strict ..
    - [ ] Run the unit tests (python -m unittest discover tests).
    - [ ] Add a smoke test: run the main script with sample_match.json and check for a successful exit code.
- [ ] Benchmark Suite:
  - [ ] Create a new test file (e.g., tests/test_benchmarks.py).
  - [ ] Create a directory with ~20 sample JSON match files with known results.
  - [ ] Write a test that iterates through these files, runs the generate_feedback function, and asserts that the number and type of
        critical/warning feedback remain consistent.

5. Dependency Hygiene

- [ ] Fix `requirements.txt`:
  - [ ] Add torch-cpu to requirements.txt to ensure a consistent CPU-only build of EasyOCR's dependency.
  - [ ] Pin all other major dependencies to their current versions.
- [ ] Fix `package.json`:
  - [ ] Initialize a proper project with npm init -y.
  - [ ] Add vite, vitest, eslint, and tailwindcss as dev dependencies.

This list provides a clear, actionable path forward. I recommend we start with the "One-week sprint checklist" from the analysis, which prioritizes
the most critical items from these lists.
