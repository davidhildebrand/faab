# faab-ol-bot: setup walkthrough

faab-ol-bot checks the FAAB optic-lobe tracking sheet once a day and posts what it finds to a Slack channel. It only
reads: the sheet (read-only access), CAVE (the default view key), and nothing is written anywhere except Slack
messages. The code is `faab/scripts/bot/faab_ol_bot.py`.

What it reports (cells are named by full root ids and tab names, never row numbers):

| check | what it means |
|---|---|
| xyz is not on the row's cell | `root_id_curr` is a current root, but the `xyz` lands on another cell (or on empty space) |
| one cell, different statuses on different rows, one of them not a known status | the same cell is listed on several rows (any tabs) with different statuses, and one is a word neither SeaTable nor the update's ranking uses (a typo, say). Differences between known statuses are not reported, because every update gives all rows of a cell its SeaTable status, or the best one (proofread > complete > adequate > review > in progress > stopped) when the cell isn't in SeaTable. Blank, NEW, NEW-DUP, and candidate are ignored |
| one cell, different people on different rows | two rows of the same cell each name someone the other row does not (names compared by initials). A row that lists everyone on the other row plus a new name is not reported, since that person saw who else was on the cell |
| root id that is not an 18-digit id | a number in a root id column that is not 18 digits, e.g. an id with a digit missing or one shown as `6.48518E+17`. Text such as block titles, group lines, and notes is skipped |
| root id stored as a number (Google Sheets keeps only 15 digits) | an id was pasted into a cell formatted as a number, so Sheets kept 15 of its 18 digits and the id now names another cell, or none. Fix: format the column as plain text and paste the id again |
| row with a root id but no usable xyz | the row has an 18-digit root id, but the xyz is empty or not three numbers |
| annotation differs from SeaTable's: which one is right? | the row's `annotation` and the SeaTable label in its own `SeaTable` column name different types, e.g. `L5` vs `ol_intrinsic / visual / Mi`. The sheet update never overwrites an annotation, so the annotator settles it |
| annotation names a type that SeaTable does not have yet: add it to SeaTable? | the annotation is more specific than SeaTable, e.g. `Tm` where SeaTable says only `ol_intrinsic / visual`, or `L2?` where SeaTable says `LMC`. An older or shorter copy of SeaTable's label (e.g. `Mi` for `ol_intrinsic / visual / Mi`) is not reported |

Stale `root_id_curr` values are not reported, because every sheet update replaces them with the root at the xyz.
Each issue is posted once. The bot remembers what it posted (`tmp/faab_ol_bot/state.json`) and lists only new issues;
the summary line always counts all open ones.

**How it runs.** Like banc-bot, it is a program that stays running inside `tmux` (or `screen`) on an always-on Ubuntu
machine. Once a day it reads the live sheet, runs the checks, and posts. It needs outgoing internet access only; no
ports have to be opened.

**When it runs: 06:30 UTC.** That is the quietest time in the CAVE edit log (2026-07-12 to 2026-11-12): the sheet's
proofreaders did none of their editing between 05:00 and 08:00 UTC in the last four weeks, and all 33 editors together
did 0.2% of theirs between 06:00 and 07:00 UTC. It also comes after the evening sheet updates (about 03:00-04:30 UTC),
and the report is waiting before the European proofreaders start (from about 08:00 UTC). The busiest time is
13:00-16:00 UTC.

The setup has six parts. Parts 1-3 happen on your Mac (try it, create the Slack app, give it read access to the sheet),
parts 4 and 5 on the Ubuntu machine (install it, keep it running), and part 6 is optional (@mentions).

---

## Part 1: try it on the Mac without Slack (2 minutes)

A dry run writes the message to a file and posts nothing. From the FAAB folder:

```bash
cd /Users/davidh/Data/Vibe/FAAB
```

```bash
venv/bin/python faab/scripts/bot/faab_ol_bot.py --book "$(ls -t faab/figures/sheets/*_Backup.xlsx | head -n 1)" --all --no-state
```

The `$(ls -t …)` part picks the newest backup in `faab/figures/sheets`. It prints the summary and the path of the message file (`tmp/faab_ol_bot/<stamp>_message.md`), which shows exactly what
would be posted. `--all` lists every open issue, and `--no-state` keeps test runs from touching the memory of what was
posted.

---

## Part 2: create the Slack app (10 minutes)

1. Open https://api.slack.com/apps and click **Create New App**, then **From scratch**.
2. App name: `faab-ol-bot`. Pick the workspace where the proofreaders are. Click **Create App**.
3. In the left menu, open **OAuth & Permissions**. Under **Scopes**, **Bot Token Scopes**, click **Add an OAuth Scope**
   and add `chat:write`. That is the only scope the bot needs.
4. Optional, under **Basic Information**, **Display Information**: an icon and a short description, for example
   "Daily checks of the FAAB optic-lobe tracking sheet".
5. Back in **OAuth & Permissions**, click **Install to Workspace** and **Allow**. If your workspace needs an admin to
   approve apps, Slack sends the request to the admin and the install finishes once they approve.
6. Copy the **Bot User OAuth Token**. It starts with `xoxb-`. Treat it like a password.
7. In Slack, pick the channel for the reports. I suggest starting with a private test channel (for example
   `#faab-ol-bot-test`), then switching to the proofreaders' channel once the messages look right. In the channel,
   type `/invite @faab-ol-bot`.
8. Get the channel id: click the channel name at the top, scroll to the bottom of the **About** tab, and copy the
   **Channel ID** (it starts with `C`).

Now store the token and the channel id in a private config file outside the repo. Create the folder:

```bash
mkdir -p ~/.config/faab-ol-bot
```

Create the file, make it readable only by you, and open it in TextEdit:

```bash
touch ~/.config/faab-ol-bot/env && chmod 600 ~/.config/faab-ol-bot/env && open -e ~/.config/faab-ol-bot/env
```

Put these two lines in it (with your values), save, and close:

```text
SLACK_BOT_TOKEN=xoxb-your-token
FAAB_OL_BOT_CHANNEL=C0123456789
```

A first real post to the test channel, from a backup (the summary goes to the channel, the details into its thread):

```bash
venv/bin/python faab/scripts/bot/faab_ol_bot.py --book "$(ls -t faab/figures/sheets/*_Backup.xlsx | head -n 1)" --post --no-state
```

(`--no-state` keeps this test from counting as "already posted", so the Ubuntu bot's first post will still list
everything.)

If the token ever leaks: Slack app page, **OAuth & Permissions**, **Revoke tokens**, then reinstall and put the new
token in the config file.

---

## Part 3: let the bot read the live sheet (15 minutes)

The bot reads the sheet through a Google "service account": a robot Google identity that you share the sheet with as
a **Viewer**, so it can read but never edit.

1. Open https://console.cloud.google.com and sign in with the Google account that owns or can share the sheet.
2. Create a project: project menu at the top, **New Project**, name `faab-ol-bot`, **Create**, then select it.
3. Turn on the Drive API: **APIs & Services**, **Library**, search **Google Drive API**, click **Enable**.
4. Create the service account: **IAM & Admin**, **Service Accounts**, **Create service account**. Name `faab-ol-bot`.
   Click **Create and continue**, skip the role and user steps, then click **Done**.
5. Make its key: click the new service account, open **Keys**, **Add key**, **Create new key**, **JSON**,
   **Create**. A `.json` file downloads. Treat it like a password.
6. Move the key next to the config file and lock it (adjust the downloaded file name):

```bash
mv ~/Downloads/faab-ol-bot-*.json ~/.config/faab-ol-bot/google-key.json
```

```bash
chmod 600 ~/.config/faab-ol-bot/google-key.json
```

7. Copy the service account's email address (it looks like `faab-ol-bot@<project-id>.iam.gserviceaccount.com`).
   In the tracking sheet, click **Share**, paste the address, choose **Viewer**, untick **Notify people**, and
   click **Share**.
8. Copy the sheet id from the sheet's URL: the part between `/d/` and `/edit`.
9. Add two lines to `~/.config/faab-ol-bot/env`:

```text
FAAB_SHEET_ID=the-id-from-the-url
GOOGLE_APPLICATION_CREDENTIALS=/Users/davidh/.config/faab-ol-bot/google-key.json
```

Test it on the Mac (a dry run from the live sheet):

```bash
venv/bin/python faab/scripts/bot/faab_ol_bot.py --live
```

If Google refuses to create the key or to share with the service account, your organization blocks it (policies
named "Disable service account key creation" or "Domain restricted sharing"). Then ask the Google Workspace admin for
an exception; until then the bot can only check backups you download (Part 2).

---

## Part 4: set up the Ubuntu machine (20 minutes)

Below, `USER@HOST` is your login on that machine.

**1. On the Ubuntu machine: Python, git, tmux, the code, and a virtual environment.** Ubuntu 24.04 comes with
Python 3.12.

```bash
sudo apt install -y git python3-venv tmux
```

```bash
mkdir -p ~/faab-ol-bot/tmp && git clone https://github.com/davidhildebrand/faab.git ~/faab-ol-bot/faab
```

```bash
python3 -m venv ~/faab-ol-bot/venv && ~/faab-ol-bot/venv/bin/python -m pip install -U pip
```

```bash
~/faab-ol-bot/venv/bin/python -m pip install -e ~/faab-ol-bot/faab openpyxl google-auth requests
```

The code puts its scratch files in `~/faab-ol-bot/tmp` (the folder next to the repository), as on the Mac.

**2. From the Mac: copy the credentials.** Make the private folders on the server:

```bash
ssh USER@HOST "mkdir -p ~/.cloudvolume/secrets ~/.config/faab-ol-bot && chmod 700 ~/.cloudvolume/secrets ~/.config/faab-ol-bot"
```

The CAVE view-only key, saved under the name the code reads (`cave-secret.json`):

```bash
scp ~/.cloudvolume/secrets/cave-secret-readonly.json USER@HOST:.cloudvolume/secrets/cave-secret.json
```

The bot's config file and the Google key:

```bash
scp ~/.config/faab-ol-bot/env ~/.config/faab-ol-bot/google-key.json USER@HOST:.config/faab-ol-bot/
```

Optional, but it saves a slow first run: the Mac's cache of sheet-point lookups.

```bash
scp /Users/davidh/Data/Vibe/FAAB/tmp/sv_at_xyz.json USER@HOST:faab-ol-bot/tmp/
```

**3. On the Ubuntu machine: lock the files and fix the key path.**

```bash
chmod 600 ~/.cloudvolume/secrets/cave-secret.json ~/.config/faab-ol-bot/env ~/.config/faab-ol-bot/google-key.json
```

```bash
nano ~/.config/faab-ol-bot/env
```

Change the `GOOGLE_APPLICATION_CREDENTIALS` line to the server's home folder, for example
`GOOGLE_APPLICATION_CREDENTIALS=/home/USER/.config/faab-ol-bot/google-key.json`, then save (Ctrl+O, Return) and quit
(Ctrl+X).

**4. Test it (a dry run from the live sheet):**

```bash
~/faab-ol-bot/venv/bin/python ~/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --live
```

The message lands in `~/faab-ol-bot/tmp/faab_ol_bot/`.

The credentials on this machine are readable only by your account, but anyone with `sudo` there could read them. Use
the view-only CAVE key only (as above), never the edit key.

---

## Part 5: keep it running in tmux (5 minutes)

`tmux` keeps a terminal session alive after you log out; you can come back to it any time. (`screen` works the same
way; its keys are given below too.)

Start a session named `faab-ol-bot`:

```bash
tmux new -s faab-ol-bot
```

Inside it, start the bot. `--serve` keeps it running and checks every day at 06:30 UTC, `--post` posts to Slack, and
`--pull` updates the code from GitHub before each check (so a change pushed to `main` is used the next morning):

```bash
~/faab-ol-bot/venv/bin/python ~/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --serve --post --pull --run-now
```

`--run-now` makes it check once right away, so you see the first post (to the test channel) without waiting for
06:30 UTC; leave it out on later starts. The bot prints a line for everything it does, for example:

```text
2026-11-15 20:10:03 UTC  faab-ol-bot serving: the check runs every day at 06:30 UTC (posts to Slack; pulls the code first); log ...; stop with Ctrl+C
2026-11-15 20:14:41 UTC  check: done
2026-11-15 20:14:41 UTC  alive; next check 2026-11-16 06:30 UTC
```

Leave the session running: press **Ctrl+B**, then **D** ("detach"). You can now log out. To look at it again:

```bash
tmux attach -t faab-ol-bot
```

The same lines are kept in a log file, which you can read without attaching:

```bash
tail -n 30 ~/faab-ol-bot/tmp/faab_ol_bot/serve.log
```

To stop the bot: attach, then press **Ctrl+C**. Start it again with the same command (without `--run-now`); it will not
post twice on the same day.

When the messages look right, switch to the proofreaders' channel: change `FAAB_OL_BOT_CHANNEL` in
`~/.config/faab-ol-bot/env` (and `/invite @faab-ol-bot` there). The next daily check uses the new channel; no restart
needed.

With `screen` instead of `tmux`: start with `screen -S faab-ol-bot`, detach with **Ctrl+A** then **D**, and come back
with `screen -r faab-ol-bot`.

**After a reboot.** tmux sessions end when the machine restarts. Either start the bot again by hand, or let the machine
start it at boot. Open your crontab:

```bash
crontab -e
```

and add this line at the end, then save:

```text
@reboot tmux new -d -s faab-ol-bot "$HOME/faab-ol-bot/venv/bin/python $HOME/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --serve --post --pull"
```

**How it behaves.**

- A failed check (for example CAVE or Google briefly down) is retried after 15, 30, and 60 minutes; after that it waits
  for the next day.
- It never posts twice on one day, also not after a restart (`tmp/faab_ol_bot/last_run.txt`).
- A day with nothing new produces no post.
- Once the Ubuntu bot is posting, do not also run `--post` from the Mac without `--no-state`: each machine keeps its
  own memory of what was posted, so both would post the same issues.

---

## Part 6 (optional): @mention the proofreaders

The bot can tag the people whose rows have issues. Make a file mapping the names (or initials) used in the sheet to
Slack member ids. A member id is on each Slack profile: click the person, the **⋮** menu, **Copy member ID**. On the
Ubuntu machine:

```bash
nano ~/.config/faab-ol-bot/people.json
```

```json
{
  "ND": "U0123ABCDEF",
  "DR": "U0456GHIJKL",
  "EB": "U0789MNOPQR"
}
```

Initials are taken from the first and last word of a name (`Nandita D` and `Nandita D.` are both `ND`). Then add a
line to `~/.config/faab-ol-bot/env` (with your user name):

```text
FAAB_OL_BOT_PEOPLE=/home/USER/.config/faab-ol-bot/people.json
```

The next daily check uses it; no restart needed.

---

## Part 7 (optional): the proofreaders' time-spent page (10 minutes)

The bot can rebuild the editing-time page (hours per person per day and week, by time of day, with a comparison table)
every day after the check, for everyone who edited in the last 4 months, and serve it from the Ubuntu machine. Nothing
is posted to Slack. The hours come from CAVE's edit log, so they are a lower bound: looking without editing is not
counted. Per-person work hours are sensitive, so the page is only served to the machine itself; you open it through an
SSH tunnel.

**Names.** The edit log has user ids. Looking up names needs your edit key, which never goes on the bot, so copy the
names file from the Mac once (and again whenever new people join, after running `scripts/admin/cave_user_names.py`
with your edit key there). On the Ubuntu machine, make the folder:

```bash
mkdir -p ~/faab-ol-bot/tmp/faab_ol_bot/activity
```

On the Mac, copy the newest names file (with your Ubuntu user and host name):

```bash
scp "$(ls -t ~/Data/Vibe/FAAB/faab/figures/misc/*_cave_user_names.csv | head -n 1)" USER@UBUNTU:faab-ol-bot/tmp/faab_ol_bot/activity/cave_user_names.csv
```

Without it, people show up as `user <id>`.

**Skip the slow first read (optional).** The first build reads 4 months of the edit log (about 300 CAVE calls at the
2-second rate limit, 10 to 15 minutes); building the page itself takes seconds, and later runs only read the new
edits. If the Mac has built the page already, copy its cache over and the Ubuntu machine's first run is quick too:

```bash
scp ~/Data/Vibe/FAAB/tmp/faab_ol_bot/activity/faab_ol_bot_ops_raw.parquet ~/Data/Vibe/FAAB/tmp/faab_ol_bot/activity/faab_ol_bot_ops_scanned.txt USER@UBUNTU:faab-ol-bot/tmp/faab_ol_bot/activity/
```

**First build.** On the Ubuntu machine, build the page once:

```bash
~/faab-ol-bot/venv/bin/python ~/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --activity-now
```

**Daily, and served.** Add `--activity --host` to the command that runs in tmux (Part 5) and to the `@reboot` line:

```bash
~/faab-ol-bot/venv/bin/python ~/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --serve --post --pull --activity --host
```

The bot prints `time-spent page served at http://127.0.0.1:8765/` and, after each daily check, `time-spent page: page
done`.

**Opening it.** On the Ubuntu machine itself, open `http://127.0.0.1:8765/` in a browser. From the Mac, open a tunnel
(leave this running in a terminal; Ctrl+C closes it):

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@UBUNTU
```

then open `http://localhost:8765/` in the Mac's browser. Leave `--bind` at its default (127.0.0.1): with
`--bind 0.0.0.0` anyone who can reach the machine on the network could read everyone's hours.

---

## Day-to-day

- Each check writes `tmp/faab_ol_bot/<stamp>_message.md` (the text) and `<stamp>_issues.tsv` (all open issues) on the
  Ubuntu machine.
- To post the full list again (for example after a sheet update), run once by hand with `--all`:

```bash
~/faab-ol-bot/venv/bin/python ~/faab-ol-bot/faab/scripts/bot/faab_ol_bot.py --live --post --all
```

- To start over, so that the next post lists every open issue as new, move `~/faab-ol-bot/tmp/faab_ol_bot/state.json`
  somewhere else.

## Later: answering messages

banc-bot answers messages instantly because it holds an open connection to Slack ("Socket Mode"; no public address
needed). The same running process can do that here, for example a direct message `check` to run the checks now, or
`check 648518347606609544` to see one cell's rows. That needs the `slack_bolt` package and, in the Slack app, Socket
Mode turned on (an app-level token with `connections:write`) and the `message.im` and `app_mention` events. The daily
check stays as it is.
