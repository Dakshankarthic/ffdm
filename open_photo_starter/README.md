# Open photograph starter set

Five openly licensed camera photographs collected from Wikimedia Commons on 2026-10-31. These are source-site original JPEGs: they have not been cropped, resized, recompressed, or AI-edited for this collection. Source provenance and camera metadata support their use as pilot source photographs; this collection does not certify that every earlier photographic adjustment is absent.

This is a small pilot set for creating and checking AI edits, not a training dataset or a measure of detector accuracy.

## Credits and licenses

### 01_kitchen.jpg

- Title: Restaurant kitchen
- Author: Raysonho @ Open Grid Scheduler / Grid Engine
- Photo date reported by source: 2015-11-27
- Source: https://commons.wikimedia.org/wiki/File:RestaurantKitchen.jpg
- License: CC0 1.0 — https://creativecommons.org/publicdomain/zero/1.0/
- Dimensions: 4928 x 3264
- Suggested pilot edit: Add or replace one small object on a counter.

### 02_living_room.jpg

- Title: Living room in Victoria, Australia
- Author: Tim Collins
- Photo date reported by source: 2010-09-21
- Source: https://commons.wikimedia.org/wiki/File:Living_Room.jpg
- License: CC BY-SA 3.0 — https://creativecommons.org/licenses/by-sa/3.0/
- Dimensions: 3600 x 2399
- Suggested pilot edit: Add or remove one small decorative object.

### 03_park_bench.jpg

- Title: Park bench
- Author: FelpMan
- Photo date reported by source: 2008-10 (source description)
- Source: https://commons.wikimedia.org/wiki/File:ParkBench.jpg
- License: Public domain (author release) — https://commons.wikimedia.org/wiki/File:ParkBench.jpg#Licensing
- Dimensions: 1290 x 846
- Suggested pilot edit: Add a small bag or bottle on the bench.

### 04_office_desk.jpg

- Title: Items on a Wikipedian's office desk
- Author: Shinealight
- Photo date reported by source: 2006-04-03
- Source: https://commons.wikimedia.org/wiki/File:Items_on_a_Wikipedian%27s_office_desk_-_20060403.jpg
- License: Public domain (author release) — https://commons.wikimedia.org/wiki/File:Items_on_a_Wikipedian%27s_office_desk_-_20060403.jpg#Licensing
- Dimensions: 3456 x 2304
- Suggested pilot edit: Replace or remove one small desk item.

### 05_street_market.jpg

- Title: Street market in Hanoi, Vietnam
- Author: Daderot
- Photo date reported by source: 2014-08-03
- Source: https://commons.wikimedia.org/wiki/File:Street_market_-_Hanoi,_Vietnam_-_DSC03457.JPG
- License: CC0 1.0 — https://creativecommons.org/publicdomain/zero/1.0/
- Dimensions: 4842 x 3262
- Suggested pilot edit: Change one small market item; avoid editing the entire scene.

The living-room image is CC BY-SA 3.0. If you distribute an adaptation, credit Tim Collins, link the source and license, identify your changes, and distribute the adaptation under the same or a compatible license. The other four photos carry CC0 or an author public-domain release. Keep the credits with any redistributed collection.

## Save each experiment

Keep the downloaded originals unchanged. Save the edited output separately, with the selected edit mask, prompt, editor/model and version, and settings. Use one small object per first experiment. Keep dimensions and alignment consistent where possible.

For a mask: white (255) identifies the intended edited region and black (0) the rest. The requested mask is not automatically an exact record of all pixels modified by the editor: inspect whether the tool also changes unselected areas. Record that behavior before treating a mask as ground truth.

For quantitative evaluation, keep each original and all its edited/transformed versions in the same data split.
