# H-0137: ライセンスの確認（2026-10-21）

同梱した 4 つのデータについて、出典のページを 2026-10-21 に開き、ライセンスの文をそのまま写した。帰属の表示は `notebooks/data/manifest.toml` の各項目の `notice` に書いた。

| ファイル | 開いたページ | ページの文（そのまま） |
|---|---|---|
| `credit-g.csv.gz` | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." 引用: "Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77." |
| `vehicle.csv.gz` | https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." 引用: "Mowforth, P. & Shepherd, B. Statlog (Vehicle Silhouettes) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HG6N." |
| `diamonds.csv.gz` | https://github.com/tidyverse/ggplot2/blob/main/LICENSE.md と https://raw.githubusercontent.com/tidyverse/ggplot2/main/LICENSE | `LICENSE.md` は "MIT License" と "Copyright (c) 2025 ggplot2 core developer team" に続く MIT の全文。`LICENSE`（CRAN の形）は "YEAR: 2025" と "COPYRIGHT HOLDER: ggplot2 core developer team"。取得した写し（seaborn-data）のリポジトリには、独自のライセンスの記載が無い。 |
| `jena_climate_2016_tail.csv.gz` | https://www.bgc-jena.mpg.de/wetter/weather_data.html | "Terms of Use (as per Creative Commons CC-BY-4.0)" / "Nutzungsbedingungen (gemäß Creative Commons CC-BY-4.0)"。リンク先は https://creativecommons.org/licenses/by/4.0/legalcode。トップページ https://www.bgc-jena.mpg.de/wetter/ に、利用条件や引用の指定は無い。 |

- OpenML の dataset 31 と 54 の `licence` の欄は、どちらも "Public" だった（`fetch_openml(...).details["licence"]`）。元の UCI のページの CC BY 4.0 に従う。
- CC BY 4.0 の 3(a) が求める表示（作成者、ライセンスへの言及と URI、保証の否認への言及、元の資料への URI、変更の有無）は、各 `notice` に書いた。
- MIT は、著作権の表示と許諾の文を写しに含めることを求める。diamonds の `notice` に、ggplot2 の `LICENSE.md` の全文を写した。
- 確認は Claude Code の WebFetch で行った。リポジトリの CI はこのページを開かない。
