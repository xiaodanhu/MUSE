# Muse: Additional Portrait Resources

## Data Files
Additional resources are included in two files: wikiart.json and wikidata.json respectively. Each json files contain a list of portraits, with each portrait presented as a dictionary of image download links and additional information. 

|Dataset  | \#portraits|
|:---------:|:---:|
|wikiart  |  25,588|
|wikidata |  26,351|

## Information Slots
Wikiart and wikidata data have different additional information slots. Some information slots can be missing for certrain portraits, which are filled with `null`. 

### Wikiart Information Slots
These are the information slots in the wikiart data:
- `title`: Title/Name of the portrait in string format
- `contentId`: Unique ID of the portrait in wikiart database
- `artistContentId`: Unique ID of the artist that create the portrait
- `artistName`: Name of the artist that create the portrait
- `yearAsString`: Year of the portrait's completion in string format
- `image`: Image download link
- `style`: Style of the portrait
- `galleryName`: Name of gallery that possess the portrait
- `description`: A brief description of the portrait

### Wikidata Information Slots
These are information slots in the wikidata:
- `names`: Titles/Names of the portrait in multiple languages. We use abbreviations for languages(e.g. `en` for English). Utf-8 codes are used.
- `sitelinks`: Links to Wikipedia page in multiple languages. An empty dictionary if none corresponding page exists.
- `id`: Wikidata entity ID of the portrait
- `description`: A brief description of the portrait
- `location`: Place of collection, usually the name of gallery
- `inception`: Time point of creation.
- `described at URL`: URLs that has descriptions about the portraits.
- `material used`: Materials used for the portrait
- `author`: Name of the artist that create the portrait
- `image_download`: Image download link
