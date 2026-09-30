# Assignment 15: step-by-step guide

Every command runs in the VS Code terminal (Terminal, New Terminal) with the
`eurosat-lora` folder open. The 📸 marks are good moments for a screenshot.

---

## Part 0: One-time setup

Check Git is installed:

```
git --version
```

Tell Git who you are (this name appears on every commit):

```
git config --global user.name "Sondos"
git config --global user.email "the-email-on-your-github-account@example.com"
```

On GitHub, create a new repository:
- Name: `eurosat-lora`
- Visibility: **Public** (Kaggle needs this to clone it without a password)
- Do **not** tick "Add a README", ".gitignore" or "license". The repo must be empty.

Copy the repository URL it shows you, like `https://github.com/YOUR_USERNAME/eurosat-lora.git`.

---

## Part 1: First upload to `main` (init, status, add, commit, push)

```
git init
git status
```
📸 `git status` lists every file as "untracked": Git sees them but is not tracking them yet.

```
git add .
git status
```
📸 Now the files are green, "changes to be committed". `git add` puts changes into the staging area, meaning "include these in the next commit".

```
git commit -m "Initial commit: LoRA project with Docker and CI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/eurosat-lora.git
git push -u origin main
```

- `git commit` saves a snapshot of the staged files with a message.
- `git branch -M main` names the current branch `main`.
- `git remote add origin ...` tells Git where your GitHub repo is and calls it `origin`.
- `git push -u origin main` uploads `main`. The `-u` remembers the link, so later you can just type `git push`.

A browser window may open asking you to sign in to GitHub. That is normal, it happens once.

**Check GitHub Actions:** open your repo, click the **Actions** tab. A run called "CI" starts.
📸 Click it: you should see **test → build → deploy**, all three jobs, because this push went to `main`.

---

## Part 2: Create the `develop` branch

```
git checkout -b develop
git push -u origin develop
```

`git checkout -b develop` creates a new branch from where you are and switches to it.

📸 In the Actions tab a new run appears for `develop`. This time **deploy is greyed out as "skipped"**. That is the branch-specific trigger working: the `if:` line in `ci.yml` only allows deploy on `main`.

---

## Part 3: `git pull`

Pull downloads commits from GitHub that you do not have locally. To have something to pull, make a change on GitHub itself:

1. On GitHub, switch the branch dropdown (top left of the file list) to **develop**.
2. Open `README.md`, click the pencil icon, add a line at the bottom such as `Edited on GitHub.`
3. Click **Commit changes**.

Back in VS Code:

```
git pull
```
📸 Git downloads the commit and your README now has the new line.

---

## Part 4: A change on `develop` (git diff, git status)

Open `configs/sanity.yaml` and change:

```yaml
debug_max_train: 50
```
to
```yaml
debug_max_train: 100
```
Save the file, then:

```
git status
git diff
```
📸 `git status` shows the file as modified. `git diff` shows exactly which line changed (red is removed, green is added).

```
git add configs/sanity.yaml
git commit -m "Sanity check: use 100 training images"
git push
```

---

## Part 5: A different change to the same line on `main`

```
git checkout main
```

Open `configs/sanity.yaml` again. Notice it says `50` here, because `main` does not have the develop commit. Change it to:

```yaml
debug_max_train: 80
```

```
git add configs/sanity.yaml
git commit -m "Sanity check: use 80 training images"
git push
```

Now both branches changed the **same line** in different ways. That is what causes a conflict.

---

## Part 6: `git diff` between the two branches

```
git diff main develop
```
📸 This shows everything that is different between the branches: the `80` vs `100` line and the README line from Part 3.

For a short summary (which files differ and how many lines):

```
git diff --stat main develop
```

Press `q` to exit if the output opens in a scrolling view.

---

## Part 7: Merge and solve the conflict

You are on `main`. Bring the develop work into it:

```
git merge develop
```

Git says:
```
CONFLICT (content): Merge conflict in configs/sanity.yaml
Automatic merge failed; fix conflicts and then commit the result.
```

```
git status
```
📸 The file is listed under "Unmerged paths" as "both modified".

Open `configs/sanity.yaml` in VS Code. You will see:

```
<<<<<<< HEAD
debug_max_train: 80
=======
debug_max_train: 100
>>>>>>> develop
```

- Between `<<<<<<< HEAD` and `=======` is the version on the branch you are on (`main`).
- Between `=======` and `>>>>>>> develop` is the version coming in from `develop`.

📸 Screenshot this. VS Code shows buttons above it. Click **Accept Incoming Change** to keep `100`. The markers disappear and only `debug_max_train: 100` remains. Save the file.

Tell Git the conflict is solved and finish the merge:

```
git add configs/sanity.yaml
git status
git commit -m "Merge develop into main, keep 100 training images"
git push
```

📸 In the Actions tab this push to `main` runs all three jobs again, including deploy.

---

## Part 8: Bring `develop` up to date

`develop` does not have the merge commit yet. Update it so both branches match:

```
git checkout develop
git merge main
git push
```

This time there is no conflict: Git just moves `develop` forward ("fast-forward").

To see the whole history as a graph:

```
git log --oneline --graph --all
```
📸 You can see the two branches splitting and joining again at the merge.

---

## Part 9 (optional): Pull request

On GitHub, go to **Pull requests → New pull request**, base `main`, compare `develop`. If there are no differences GitHub will say so. Make a small change on `develop` first (for example, a line in the README), push it, then open the pull request. The CI runs **test** and **build** on the pull request, and shows a green tick before you are allowed to merge.

---

## Part 10: See the published Docker image

On your repo's main page, the right side has a **Packages** section with `eurosat-lora`. That is the image the deploy job pushed. Its full name is:

```
ghcr.io/YOUR_USERNAME/eurosat-lora:latest
```

---

## Part 11: Real training on Kaggle

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and upload `kaggle/kaggle_run.ipynb`.
2. In the right panel: turn on **GPU** and **Internet**, and use **Add Input** to attach the `eurosat-rgb-dataset` dataset.
3. In the first code cell, replace `YOUR_USERNAME` with your GitHub username.
4. Run all cells. The last cell shows the results table.

---

## Quick reference

| Command | What it does |
|---|---|
| `git status` | what changed and what is staged |
| `git add <file>` / `git add .` | stage changes for the next commit |
| `git commit -m "..."` | save a snapshot of the staged changes |
| `git push` | upload commits to GitHub |
| `git pull` | download commits from GitHub |
| `git checkout -b <name>` | create a branch and switch to it |
| `git checkout <name>` | switch branch |
| `git diff` | unstaged changes in your files |
| `git diff main develop` | differences between two branches |
| `git merge <branch>` | bring another branch's commits into this one |
| `git log --oneline --graph --all` | history of all branches as a graph |
