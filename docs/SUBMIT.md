# Submission checklist

**Deadline:** 31 October 2026, 11:59 PM Pacific = **1 November 2026, 2:59 PM Malaysia time**. Aim to submit by 30 October.

## 1. Put the code on GitHub (about 5 minutes)

1. Go to https://github.com/new and create a **public** repository named `squadro-match-lab`. Do not add a README, license or .gitignore (the folder already has them).
2. In Terminal on your Mac:

   ```bash
   cd "~/Documents/icloud doc/Hacks/AGX Sports analytics/squadro-match-lab"
   git remote add origin https://github.com/ShermaineYap/squadro-match-lab.git
   git push -u origin main
   ```

   If you use GitHub Desktop instead: File → Add Local Repository → choose the `squadro-match-lab` folder → Publish repository, and untick "Keep this code private".

## 2. Turn on the live site (about 2 minutes)

1. On the repository page: **Settings → Pages**.
2. Source: **Deploy from a branch**. Branch: **main**, folder **/ (root)**. Save.
3. Wait 1–2 minutes, then open https://shermaineyap.github.io/squadro-match-lab/ and check the Predict view loads.
4. Open the **Evidence** view and confirm the reproduction check says **Exact match**.

If your GitHub username is not `ShermaineYap`, replace it in the two links in `README.md` and `docs/DEVPOST.md`.

## 3. Submit on Devpost (about 10 minutes)

1. https://aqxanalyticsfour.devpost.com → **Join hackathon** (if you have not) → **Submit project**.
2. Copy each section of `docs/DEVPOST.md` into the matching field.
3. Add the GitHub link and the live demo link.
4. Upload images from `docs/screenshots/`. Use `predict.png` as the thumbnail.
5. Optional but recommended: record the 2-minute demo from `docs/VIDEO_SCRIPT.md` and add the YouTube link.

## 4. Before you press submit

- [ ] Live link opens on your phone.
- [ ] Repository is public and the README shows the screenshot.
- [ ] Devpost description mentions the three judging criteria: practical application (coach actions), analytical insight (out-of-sample findings with intervals), data presentation (explanations and charts).
- [ ] Submitted before 1 Nov, 2:59 PM Malaysia time.
