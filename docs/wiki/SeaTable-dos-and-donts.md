# SeaTable dos and don'ts

## Overview

We use SeaTable to annotate cell types and keep track of who is working on what. SeaTable is powerful but not always user-friendly, and every change you make, even to a view's sort order or filters, affects all other users immediately. The web interface also does not handle large copy/paste operations.

**Before you make any edit, or even change a view,** read the [SeaTable section of the README](README#seatable). And please ask whenever you are not sure: if you think you should add a cell or change an entry but are not quite sure what details to fill in, we would rather help the first few times while you get up to speed than fix errors later.

## Do

- **Work in your own view.** Create a view for yourself in the "TeamOL" group and set your filters, sort order, and hidden columns there.
- **Check before you add.** Look up the cell's current root_id at its point in Neuroglancer, then search for it in SeaTable. Each neuron gets one row. See [Checking whether a cell is already in SeaTable](README#checking-whether-a-cell-is-already-in-seatable).
- **Add new rows at the bottom of the unfiltered table,** with at least point_xyz, side, superclass, status, and your initials as annotator.
- **Put point_xyz on the backbone** (a large profile, not the soma), picked in a 2D panel and confirmed by hovering. See [Choosing point_xyz](README#choosing-point_xyz).
- **Sign your work.** Add your initials as annotator when you annotate and as proofreader when you merge or cleave. Start notes with your initials and separate entries with semicolons.
- **Keep it small.** Work with about 5 rows or fewer at a time. If you need more than that, reach out.
- **Contact the person listed** before changing someone else's annotation.
- **Mark it in our sheet.** After you add a cell to SeaTable, write "added" in that row's SeaTable column of the optic lobe targets sheet.

## Don't

- **Don't change the default view** (no sorting, filtering, or hiding columns), or anyone else's view.
- **Don't paste or edit many rows at once** in the web interface. Bulk changes are done programmatically, with a dry run, by someone experienced.
- **Don't type into root_id, serial_id, or supervoxel_id.** They are filled in automatically from point_xyz (root_id about every 30 minutes).
- **Don't put point_xyz on the soma,** a thin twig, or the edge of a profile.
- **Don't add a cell that is already there.** Root IDs change with every edit, so search by the cell's current ID; a cell edited in the last half hour may still show its previous ID.
- **Don't delete rows** while you are getting started. Mark a repeated row status = "duplicate" and let us know.
- **Don't guess.** If you are unsure what to fill in, ask first.

## Questions

Message David Hildebrand on Slack, or post on #aedes-proofreading.
