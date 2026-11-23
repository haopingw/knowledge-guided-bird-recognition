# Knowledge check against dev annotations

For every feature -> CUB attribute mapping: positive rate (certainty >= 3) in the species' own dev images vs. the mean over the other 19 species. `ok` = own rate higher than others.

Supported: 75/98 mappings.

| sid | species | feature | CUB attribute | own | others | check |
|---|---|---|---|---|---|---|
| S01 | American Pipit | Thin, pointed, warbler-like bill, not a thick seed | has_bill_shape::needle | 0.00 | 0.00 | **check** |
| S01 | American Pipit | Thin, pointed, warbler-like bill, not a thick seed | has_bill_shape::all-purpose | 0.82 | 0.16 | ok |
| S01 | American Pipit | Plain gray-brown back with little or no streaking | has_back_pattern::solid | 0.00 | 0.10 | **check** |
| S01 | American Pipit | Buffy underparts with fine streaks on the breast | has_breast_pattern::striped | 0.55 | 0.29 | ok |
| S01 | American Pipit | Buffy underparts with fine streaks on the breast | has_breast_color::buff | 0.70 | 0.47 | ok |
| S01 | American Pipit | Pale eyebrow and pale eyering on a plain face | has_head_pattern::eyebrow | 0.21 | 0.13 | ok |
| S02 | Baird Sparrow | Necklace of short black streaks across a pale uppe | has_breast_pattern::striped | 0.52 | 0.29 | ok |
| S02 | Baird Sparrow | Ochre-buff face and broad ochre central crown stri | has_head_pattern::striped | 0.60 | 0.25 | ok |
| S02 | Baird Sparrow | Two thin dark whisker marks on each side of the th | has_head_pattern::malar | 0.15 | 0.19 | **check** |
| S02 | Baird Sparrow | Scaly-looking back with buff feather edges | has_back_pattern::striped | 0.50 | 0.60 | **check** |
| S03 | Brewer Sparrow | Finely streaked brown crown with no pale central s | has_crown_color::brown | 0.57 | 0.47 | ok |
| S03 | Brewer Sparrow | Crisp complete white eyering on a plain gray-brown | has_head_pattern::eyering | 0.27 | 0.16 | ok |
| S03 | Brewer Sparrow | Plain, unstreaked pale gray breast | has_breast_pattern::solid | 0.85 | 0.41 | ok |
| S03 | Brewer Sparrow | Plain, unstreaked pale gray breast | has_breast_color::grey | 0.53 | 0.31 | ok |
| S03 | Brewer Sparrow | Overall very plain and pale, with a small pinkish  | has_bill_shape::cone | 0.80 | 0.79 | ok |
| S04 | Chipping Sparrow | Bright rusty-red cap on the crown | has_crown_color::rufous | 0.14 | 0.01 | ok |
| S04 | Chipping Sparrow | Bright rusty-red cap on the crown | has_head_pattern::capped | 0.14 | 0.10 | ok |
| S04 | Chipping Sparrow | Black eye stripe running from the bill through the | has_head_pattern::eyeline | 0.41 | 0.13 | ok |
| S04 | Chipping Sparrow | Clean white eyebrow above the eye stripe | has_head_pattern::eyebrow | 0.14 | 0.13 | ok |
| S04 | Chipping Sparrow | Plain, unstreaked pale gray breast and nape | has_breast_pattern::solid | 0.70 | 0.42 | ok |
| S04 | Chipping Sparrow | Plain, unstreaked pale gray breast and nape | has_breast_color::grey | 0.39 | 0.32 | ok |
| S05 | Clay colored Sparrow | Brown crown with a thin pale central stripe | has_head_pattern::striped | 0.29 | 0.27 | ok |
| S05 | Clay colored Sparrow | Brown crown with a thin pale central stripe | has_crown_color::brown | 0.50 | 0.47 | ok |
| S05 | Clay colored Sparrow | Brown cheek patch outlined in dark, with pale lore | has_head_pattern::eyebrow | 0.19 | 0.13 | ok |
| S05 | Clay colored Sparrow | Contrasting plain gray nape | has_nape_color::grey | 0.48 | 0.37 | ok |
| S05 | Clay colored Sparrow | Unstreaked buffy-white breast | has_breast_pattern::solid | 0.77 | 0.41 | ok |
| S05 | Clay colored Sparrow | Unstreaked buffy-white breast | has_breast_color::buff | 0.48 | 0.48 | **check** |
| S06 | House Sparrow | Male: black bib on the throat and upper breast | has_throat_color::black | 0.27 | 0.15 | ok |
| S06 | House Sparrow | Male: gray crown with chestnut sides of the head a | has_crown_color::grey | 0.43 | 0.20 | ok |
| S06 | House Sparrow | Male: gray crown with chestnut sides of the head a | has_nape_color::rufous | 0.05 | 0.01 | ok |
| S06 | House Sparrow | Female: plain buff-brown with a pale buffy stripe  | has_head_pattern::eyebrow | 0.18 | 0.13 | ok |
| S06 | House Sparrow | Thick conical bill; back streaked black and buff | has_bill_shape::cone | 1.00 | 0.78 | ok |
| S06 | House Sparrow | Thick conical bill; back streaked black and buff | has_back_pattern::striped | 0.65 | 0.59 | ok |
| S07 | Field Sparrow | Bright pink bill | has_bill_color::pink | 0.04 | 0.01 | ok |
| S07 | Field Sparrow | Plain gray face with a bold white eyering | has_head_pattern::eyering | 0.14 | 0.17 | **check** |
| S07 | Field Sparrow | Rusty crown and a thin rusty line behind the eye | has_crown_color::rufous | 0.00 | 0.02 | **check** |
| S07 | Field Sparrow | Unstreaked buffy-gray breast | has_breast_pattern::solid | 0.71 | 0.42 | ok |
| S08 | Grasshopper Sparrow | Plain, unstreaked buffy breast (adults) | has_breast_pattern::solid | 0.53 | 0.43 | ok |
| S08 | Grasshopper Sparrow | Plain, unstreaked buffy breast (adults) | has_breast_color::buff | 0.71 | 0.47 | ok |
| S08 | Grasshopper Sparrow | Flat head with a pale central crown stripe | has_head_pattern::striped | 0.33 | 0.27 | ok |
| S08 | Grasshopper Sparrow | Large flat-looking bill and short, spiky tail | has_bill_shape::cone | 0.64 | 0.80 | **check** |
| S09 | Harris Sparrow | Black crown, face and bib (breeding) or black blot | has_throat_color::black | 0.71 | 0.12 | ok |
| S09 | Harris Sparrow | Black crown, face and bib (breeding) or black blot | has_crown_color::black | 0.85 | 0.31 | ok |
| S09 | Harris Sparrow | Bright pink bill | has_bill_color::pink | 0.00 | 0.01 | **check** |
| S09 | Harris Sparrow | Large sparrow with whitish belly and streaked flan | has_belly_color::white | 0.78 | 0.51 | ok |
| S10 | Henslow Sparrow | Olive-green head and nape | has_nape_color::olive | 0.00 | 0.00 | **check** |
| S10 | Henslow Sparrow | Rusty-red wings | has_wing_color::rufous | 0.11 | 0.02 | ok |
| S10 | Henslow Sparrow | Fine dark streaks across a buffy breast and flanks | has_breast_pattern::striped | 0.37 | 0.30 | ok |
| S10 | Henslow Sparrow | Large pale bill and short tail | has_bill_shape::cone | 0.76 | 0.79 | **check** |
| S11 | Le Conte Sparrow | Bright orange-buff face and eyebrow framing a gray | has_head_pattern::eyebrow | 0.10 | 0.14 | **check** |
| S11 | Le Conte Sparrow | Pinkish-chestnut nape with fine streaks | has_nape_color::rufous | 0.00 | 0.01 | **check** |
| S11 | Le Conte Sparrow | White central crown stripe between dark stripes | has_head_pattern::striped | 0.45 | 0.26 | ok |
| S11 | Le Conte Sparrow | Orange-buff breast with thin streaks along the fla | has_breast_color::buff | 0.62 | 0.47 | ok |
| S12 | Lincoln Sparrow | Crisp, fine black streaks on a buffy breast band | has_breast_pattern::striped | 0.63 | 0.29 | ok |
| S12 | Lincoln Sparrow | Crisp, fine black streaks on a buffy breast band | has_breast_color::buff | 0.67 | 0.47 | ok |
| S12 | Lincoln Sparrow | Gray face and gray eyebrow with a buffy whisker st | has_head_pattern::eyebrow | 0.14 | 0.13 | ok |
| S12 | Lincoln Sparrow | Gray face and gray eyebrow with a buffy whisker st | has_head_pattern::malar | 0.09 | 0.19 | **check** |
| S12 | Lincoln Sparrow | Brown crown with a gray central stripe | has_head_pattern::striped | 0.41 | 0.26 | ok |
| S12 | Lincoln Sparrow | Clean whitish belly | has_belly_color::white | 0.32 | 0.53 | **check** |
| S13 | Nelson Sharp tailed Sparrow | Gray nape and gray central crown stripe | has_nape_color::grey | 0.40 | 0.38 | ok |
| S13 | Nelson Sharp tailed Sparrow | Blurry streaks on an orange-buff breast and flanks | has_breast_color::buff | 0.83 | 0.46 | ok |
| S13 | Nelson Sharp tailed Sparrow | White throat and white belly | has_throat_color::white | 0.15 | 0.46 | **check** |
| S14 | Savannah Sparrow | Yellow eyebrow in front of and over the eye | has_head_pattern::eyebrow | 0.29 | 0.13 | ok |
| S14 | Savannah Sparrow | Fine dark streaks on the white breast and flanks | has_breast_pattern::striped | 0.73 | 0.28 | ok |
| S14 | Savannah Sparrow | Fine dark streaks on the white breast and flanks | has_breast_color::white | 0.70 | 0.46 | ok |
| S14 | Savannah Sparrow | Pale central crown stripe | has_head_pattern::striped | 0.52 | 0.26 | ok |
| S14 | Savannah Sparrow | Small bill and short, notched tail | has_tail_shape::notched_tail | 0.65 | 0.57 | ok |
| S15 | Song Sparrow | Heavy brown streaks on the breast merging into a c | has_breast_pattern::striped | 0.58 | 0.29 | ok |
| S15 | Song Sparrow | Heavy brown streaks on the breast merging into a c | has_breast_color::brown | 0.79 | 0.22 | ok |
| S15 | Song Sparrow | Thick dark whisker (malar) stripe beside the white | has_head_pattern::malar | 0.24 | 0.18 | ok |
| S15 | Song Sparrow | Broad gray eyebrow and brown stripe behind the eye | has_head_pattern::eyebrow | 0.24 | 0.13 | ok |
| S15 | Song Sparrow | Streaked brown back and long rounded tail | has_back_pattern::striped | 0.65 | 0.59 | ok |
| S15 | Song Sparrow | Streaked brown back and long rounded tail | has_tail_shape::rounded_tail | 0.07 | 0.11 | **check** |
| S16 | Tree Sparrow | Unstreaked gray breast with a dark central spot | has_breast_pattern::solid | 0.90 | 0.41 | ok |
| S16 | Tree Sparrow | Unstreaked gray breast with a dark central spot | has_breast_color::grey | 0.32 | 0.32 | **check** |
| S16 | Tree Sparrow | Bicolored bill: dark upper, yellow lower mandible | has_bill_color::yellow | 0.32 | 0.02 | ok |
| S16 | Tree Sparrow | Rusty crown and rusty line behind the eye on a gra | has_crown_color::rufous | 0.27 | 0.01 | ok |
| S16 | Tree Sparrow | Two white wingbars on rusty wings | has_wing_pattern::multi-colored | 0.35 | 0.22 | ok |
| S17 | Vesper Sparrow | White outer tail feathers | has_upper_tail_color::white | 0.31 | 0.15 | ok |
| S17 | Vesper Sparrow | Bold white eyering | has_head_pattern::eyering | 0.27 | 0.16 | ok |
| S17 | Vesper Sparrow | Finely streaked breast and dark-bordered ear patch | has_breast_pattern::striped | 0.43 | 0.30 | ok |
| S18 | White crowned Sparrow | Bold black-and-white stripes on the crown (brown-a | has_head_pattern::striped | 0.25 | 0.27 | **check** |
| S18 | White crowned Sparrow | Bold black-and-white stripes on the crown (brown-a | has_crown_color::black | 0.67 | 0.32 | ok |
| S18 | White crowned Sparrow | Bold black-and-white stripes on the crown (brown-a | has_crown_color::white | 0.56 | 0.13 | ok |
| S18 | White crowned Sparrow | Plain, unstreaked gray breast and face | has_breast_pattern::solid | 0.79 | 0.41 | ok |
| S18 | White crowned Sparrow | Plain, unstreaked gray breast and face | has_breast_color::grey | 0.68 | 0.30 | ok |
| S18 | White crowned Sparrow | Pink, orange or yellowish bill | has_bill_color::pink | 0.00 | 0.01 | **check** |
| S18 | White crowned Sparrow | Pink, orange or yellowish bill | has_bill_color::orange | 0.38 | 0.10 | ok |
| S18 | White crowned Sparrow | No white throat patch; the throat is gray | has_throat_color::grey | 0.70 | 0.28 | ok |
| S19 | White throated Sparrow | Sharply outlined bright white throat patch | has_throat_color::white | 0.72 | 0.43 | ok |
| S19 | White throated Sparrow | Black-and-white or brown-and-tan striped crown | has_head_pattern::striped | 0.18 | 0.27 | **check** |
| S19 | White throated Sparrow | Dark gray bill and gray, lightly blurred breast | has_breast_color::grey | 0.78 | 0.30 | ok |
| S20 | Bank Swallow | Distinct brown band across a white breast | has_breast_color::brown | 0.25 | 0.25 | ok |
| S20 | Bank Swallow | Distinct brown band across a white breast | has_breast_pattern::multi-colored | 0.25 | 0.17 | ok |
| S20 | Bank Swallow | Plain brown upperparts and white throat | has_throat_color::white | 0.63 | 0.43 | ok |
| S20 | Bank Swallow | Plain brown upperparts and white throat | has_back_color::brown | 0.53 | 0.60 | **check** |
| S20 | Bank Swallow | Tiny bill and long, pointed wings | has_wing_shape::pointed-wings | 0.27 | 0.25 | ok |
| S20 | Bank Swallow | Slightly notched tail | has_tail_shape::notched_tail | 0.19 | 0.60 | **check** |