# SeaTable dos and don'ts

## Overview

We use SeaTable to annotate cell types and keep track of who is working on what. SeaTable is powerful but not always user-friendly, and every change you make, even to a view's sort order or filters, affects all other users immediately. The web interface also does not handle large copy/paste operations.

**Before you make any edit, or even change a view,** read the [SeaTable section of the README](README#seatable). And please ask whenever you are not sure: if you think you should add a cell or change an entry but are not quite sure what details to fill in, we would rather help the first few times while you get up to speed than fix errors later.

## Do

- **Work in your own view.** Create a view for yourself in the "TeamOL" group and set your filters, sort order, and hidden columns there.
- **Check before you add.** Look up the cell's current root_id at its point in Neuroglancer, then search for it in SeaTable. Each neuron gets one row. See [Checking whether a cell is already in SeaTable](README#checking-whether-a-cell-is-already-in-seatable).
- **Add new rows at the bottom of the unfiltered table,** with at least point_xyz, side, superclass, status, and your initials as annotator.
- **Put point_xyz on the backbone, in clear cytoplasm:** a large profile, away from mitochondria, vesicle clusters, membranes, and the soma, picked in a 2D panel. Before you save, click there and check that your neuron gets selected. See [Choosing point_xyz](README#choosing-point_xyz).
- **Sign your work in the right column.** annotator = who labeled the neuron; proofreader = who merged or cleaved it. If you did both, put your initials in both. Start notes with your initials and separate entries with semicolons.
- **Check the bad supervoxel view now and then.** It lists rows whose point maps to no segment (root_id and supervoxel_id 0), usually a point on a mitochondrion.
- **Keep it small.** Work with about 5 rows or fewer at a time. If you need more than that, reach out.
- **Contact the person listed** before changing someone else's annotation.
- **Mark it in our sheet.** After you add a cell to SeaTable, write "added" in that row's SeaTable column of the optic lobe targets sheet.

## Don't

- **Don't change the default view** (no sorting, filtering, or hiding columns), or anyone else's view.
- **Don't paste or edit many rows at once** in the web interface. Bulk changes are done programmatically, with a dry run, by someone experienced.
- **Don't type into root_id, serial_id, or supervoxel_id.** Leave them blank: the scripts fill them in from point_xyz about every 30 minutes.
- **Don't put point_xyz on the soma,** a mitochondrion, a thin twig, or the edge of a profile.
- **Don't add empty rows.** Each one uses up a serial_id (we want to keep these numbers small) and counts as a bad supervoxel row, which causes glitches downstream. If you add one by accident, delete it.
- **Don't add a cell that is already there.** Root IDs change with every edit, so search by the cell's current ID; a cell edited in the last half hour may still show its previous ID.
- **Don't delete other people's rows.** If two rows of the same neuron are both yours, cross-check the annotations and delete one; if they came from different people, mark one status = "duplicate" and let us know.
- **Don't guess.** If you are unsure what to fill in, ask first.

## Questions

Message David Hildebrand on Slack, or post on #aedes-proofreading.
