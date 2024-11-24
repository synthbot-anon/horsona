#!/bin/bash

# Check if project name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME=$1

# Copy boilerplate to new project directory
cp -r ./boilerplate "./$PROJECT_NAME"
cd "./$PROJECT_NAME"

# Remove poetry.lock if it exists
rm -f poetry.lock

# Get git user info
GIT_NAME=$(git config user.name)
GIT_EMAIL=$(git config user.email)

# Update pyproject.toml with new project name and author
sed -i "s/name = \"boilerplate\"/name = \"$PROJECT_NAME\"/" pyproject.toml
sed -i "s/\"Synthbot Anon <synthbot.anon@gmail.com>\"/\"$GIT_NAME <$GIT_EMAIL>\"/" pyproject.toml
sed -i "s/boilerplate/$PROJECT_NAME/" README.md
sed -i "s/Boilerplate/$PROJECT_NAME/" README.md

# Rename src directory
mv src/boilerplate "src/$PROJECT_NAME"

echo "Project $PROJECT_NAME created successfully!"
