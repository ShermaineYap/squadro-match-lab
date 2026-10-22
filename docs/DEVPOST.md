# Devpost submission text

Paste each section into the matching Devpost field.

---

**Project name:** Squadro Match Lab

**Tagline (max ~200 characters):**
Audited badminton win probabilities a coach can explain: any BWF fixture, why the model leans, a live update after game 1, draw odds and six findings from 10,852 unseen matches.

**Links**
* Live demo: https://shermaineyap.github.io/squadro-match-lab/
* Source code: https://github.com/ShermaineYap/squadro-match-lab

**Built with:** python, pandas, scikit-learn, numpy, glicko-2, logistic-regression, javascript, html, css, svg, github-pages

---

## Inspiration

I play and follow badminton, and for my final-year project I scraped five seasons of BWF World Tour results and built a match-prediction model. It worked, but it lived behind a Python server and a long report. A coach preparing for next week's tournament cannot use either. I wanted to turn that research into something a coach, analyst or player could open on a phone and act on.

## What it does

Squadro Match Lab answers the questions a coach asks before and during a tournament.

* **Predict:** pick any two players or pairs in any of the five disciplines and get a win probability, plus the seven factors behind it in plain English (rating gap, recent form, career record, experience, head-to-head, rest days, rating certainty). Each factor shows how far the probability would move without it.
* **Live after game 1:** enter the game-1 score and the forecast updates. On all 3,822 matches of 2026, which this model never saw, ROC-AUC rises from 0.798 before the match to 0.901 after game 1.
* **Match context:** optional adjustments for home advantage and three-game fatigue, two effects the base model cannot see.
* **Draw odds:** exact title odds for a 16-entry knockout and a "draw luck" view that shows how much the draw alone moves each contender.
* **Scout:** power rankings, rating history since 2021, form, and 2026 wins above expectation, which flags players whose rating has not caught up with their results.
* **Insights and Evidence:** six findings from 10,852 out-of-sample matches, each with a 95% interval and a coaching action, plus the full audit and a live reproducibility check.

### Six findings for coaches

1. Home players win **3.3 points** more often than the model expects (95% CI +0.4 to +6.1).
2. Coming off a three-game match costs **3.1 points** of win probability next round (CI −5.0 to −1.1).
3. **82.5%** of game-1 winners take the match; an underdog who wins game 1 wins 57.6% of the time.
4. Men's singles is the hardest event to call (ROC-AUC 0.73; one favourite in three loses). Women's doubles is the easiest (0.86).
5. Upsets peak in finals: favourites win 74% in the round of 32 but 65% in finals.
6. Under a 50-point Glicko-2 gap the stronger side wins 51%; above 450 points, 93%.

## How I built it

* **Data:** 29,792 completed World Tour matches (Jan 2021 to Aug 2026), collected with my open-source scraper BWFWebScrap and cleaned in my FYP pipeline.
* **Model:** an order-symmetric logistic regression over Elo, Glicko-2, form, experience, rest and head-to-head features. It scores each fixture both ways round and averages in log-odds, so P(A beats B) = 1 − P(B beats A) exactly. Trained on 2022–2024 and frozen before 2025.
* **Making it portable:** because of that symmetry, the intercept and every side-invariant input cancel algebraically, leaving a short weighted sum of differences. I extract those weights from the scikit-learn bundle and assert that the portable model reproduces the served model on all 10,852 audit matches, to a largest difference of 3.3 × 10⁻¹⁶. The app re-runs the check in the browser.
* **New analyses:** residual analyses against the frozen model's out-of-sample predictions, with tournament-cluster bootstrap intervals; an in-match model fitted on 2025 and tested on 2026; an exact dynamic-programming bracket calculator.
* **App:** a single static page (HTML, CSS, vanilla JavaScript, hand-drawn SVG charts) served from GitHub Pages. No server and no keys.

## Challenges I ran into

* **A fatigue effect that was too good.** My first "second match of the same day" analysis showed a +29-point effect. It was leakage: match IDs within a day are not in round order, so a "previous" match was sometimes a later one. Ordering by round fixed it, and the effect became inconclusive (126 matches), so I dropped it. The three-game fatigue effect survived the fix.
* **Proving the browser model is the real model.** Rounding the exported ratings produced differences of 8 × 10⁻⁵. Exporting full precision brought the live check down to 2.2 × 10⁻¹⁶.
* **Separating a real effect from recalibration.** Adding home and fatigue terms improved 2026 log loss, but most of that gain came from rescaling the model's margin. I report the effects with their intervals and keep the adjustment optional instead of claiming a better model.

## Accomplishments that I'm proud of

* Every probability in the app can be traced to its factors and reproduced exactly from the frozen model.
* The findings are measured on seasons the model never saw, with honest intervals, and each one ends in an action a coach can take.
* It runs instantly on a phone with no backend.

## What I learned

Statistical honesty is a feature. The most useful thing to show a coach is often not a single number but how sure that number is: the men's singles caveat, the draw-luck range, the effect with its interval.

## What's next for Squadro Match Lab

* Refresh ratings weekly from the scraper so forecasts track the live tour.
* Load real tournament draws as soon as BWF publishes them and compute everyone's path to the title.
* A point-by-point live model for broadcasters and bench coaches.
* Calibrate probabilities per discipline before using them for operational decisions.
