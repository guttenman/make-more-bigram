from names_dataset import NameDataset
import random
import re

#prioritise english language names for predictable outcome
common_names = """
Aaron Abigail Adam Alexander Alice Amelia Amy Andrew Anna
Anthony Arthur Ava Benjamin Bethany Blake Brandon Brian
Callum Cameron Chloe Christopher Daniel David Daisy Dylan
Edward Eleanor Elizabeth Ella Emily Emma Ethan Eva Evelyn
Finley Florence Freddie George Grace Harry Henry Holly
Hugo Isabella Jack Jacob James Jasmine Jessica John Joseph
Joshua Julia Kai Katie Leo Lewis Lily Logan Lucas Luke
Madeleine Mia Michael Millie Molly Nathan Noah Oliver Olivia
Oscar Owen Poppy Rachel Rebecca Riley Robert Rosie Ruby
Samuel Sarah Sebastian Sophia Sophie Thomas Toby Victoria
William Zachary
""".split()

print("load names")
nd = NameDataset()


def clean_name(name):
    name = name.strip()

    #ASCII letters only, min. 2 characters
    if not re.fullmatch(r"[A-Za-z]{2,}", name):
        return None

    #at least one standard vowel
    if not re.search(r"[aeiou]", name, re.IGNORECASE):
        return None

    #no 3 or more identical consecutive letters
    if re.search(r"(.)\1{2,}", name, re.IGNORECASE):
        return None

    #no names longer than 15 characters
    if len(name) > 15:
        return None

    #no unusual letter combinations
    suspicious_patterns = [
        r"xj", r"jx", r"qj", r"jq",
        r"zx", r"xz", r"wq",
    ]

    if any(
        re.search(pattern, name, re.IGNORECASE)
        for pattern in suspicious_patterns
    ):
        return None

    return name


#clean the full first-name dataset
all_names = {
    cleaned
    for name in nd.first_names.keys()
    if (cleaned := clean_name(name))
}

print(f"names after filtering: {len(all_names)}")


#get UK names
print("get UK names")

uk_top = nd.get_top_names(
    n=5000,
    country_alpha2="GB",
    use_first_names=True
)

uk_names = set()

if uk_top and "GB" in uk_top:
    for gender_names in uk_top["GB"].values():
        if isinstance(gender_names, list):
            uk_names.update(gender_names)


#clean and combine UK names with curated names
priority_names = {
    cleaned
    for name in uk_names | set(common_names)
    if (cleaned := clean_name(name))
}

#ensure priority names are included, even if absent from the full first name dictionary
selected_names = list(priority_names)

#only use UK-associated names and curated names
if len(selected_names) < 50_000:
    other_names = list(all_names - priority_names)
    random.shuffle(other_names)

    remaining = 50_000 - len(selected_names)
    selected_names.extend(other_names[:remaining])

#50k names
if len(selected_names) < 40_000:
    print(
        f"only {len(selected_names)} names found. "
        "not enough names."
    )
elif len(selected_names) > 50_000:
    selected_names = selected_names[:50_000]

#randomise final order
random.shuffle(selected_names)

#1 name per line
with open("names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(selected_names) + "\n")

print(f"saved {len(selected_names)} names to names.txt")
print(f"prioritised UK names: {len(priority_names)}")
