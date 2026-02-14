EcoBenchmark:

- Benchmark preformance of leading LLMs for ecological detection
- Compare capabilities of LLM image recognition to a tree detection model, deepforest
- Measure the impacts of adding training data, tweaking prompts 



# 1. Initialize Git
git init

# 2. Create a smart .gitignore file
# This prevents you from accidentally uploading your 50m x 50m images or keys
echo "data/" >> .gitignore
echo "venv/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "*.ipynb_checkpoints" >> .gitignore

# 3. Add your source code
git add src/ .gitignore requirements.txt README.md

# 4. Commit locally
git commit -m "Initial commit: Benchmark architecture and preprocessing"

# 5. Create the repo on GitHub (requires GitHub CLI)
# If you don't have 'gh', just create a "New Repo" on github.com and follow the "push an existing repository" instructions.
gh repo create ecobenchmark --public --source=. --remote=origin
git push -u origin main