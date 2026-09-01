import html
import json
import os
import pickle
import re

import jsonlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy

nlp = spacy.load(
    "en_core_web_lg",
    disable=[
        "parser",
        "senter",
        "ner",
        "entity_ruler",
        "textcat",
        "morphologizer",
        "trainable_lemmatizer",
    ],
)


def makerecallplot(data, type="concept"):
    # Prepare long-form DataFrame for seaborn (model, recall)
    recall_rows = []
    for model, values in data.items():
        for _, r in values:
            name = model.replace("-20241022", "").replace("-20250514", "")
            recall_rows.append({"model": name, "recall": 1 - r})
    recall_df = pd.DataFrame(recall_rows)
    # Create violin plot
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="model", y="recall", data=recall_df, inner="quartile", palette="Set2"
    )
    plt.title(f"Recall Distribution per Model ({type})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig.savefig(f"{type}_recall.png")


def makeprecisionplot(data, type="concept"):
    # Prepare long-form DataFrame for seaborn (model, recall)
    precision_rows = []
    for model, values in data.items():
        for p, _ in values:
            name = model.replace("-20241022", "").replace("-20250514", "")
            precision_rows.append({"model": name, "precision": p})
    precision_df = pd.DataFrame(precision_rows)
    # Create violin plot
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="model", y="precision", data=precision_df, inner="quartile", palette="Set2"
    )
    plt.title(f"Precision Distribution per Model ({type})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig.savefig(f"{type}_precision.png")


def makef1plot(data, type="concept"):
    # Prepare long-form DataFrame for seaborn (model, recall)
    f1_rows = []
    for model, values in data.items():
        for p, r in values:
            name = model.replace("-20241022", "").replace("-20250514", "")
            f1 = 2 * ((p * r) / (p + r))
            f1_rows.append({"model": name, "f1": f1})
    f1_df = pd.DataFrame(f1_rows)
    # Create violin plot
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x="model", y="f1", data=f1_df, inner="quartile", palette="Set2")
    plt.title(f"F1 Distribution per Model ({type})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig.savefig(f"{type}_f1.png")


citations = re.compile(r"\[[\d]+\]")


def getcites(summary):
    cites = set()
    matches = [
        int(m.replace("[", "").replace("]", "")) for m in citations.findall(summary)
    ]
    [cites.add(m - 1) for m in matches]
    return list(cites)


def getnouns(text):
    nouns = set()
    text = html.unescape(re.sub(r"<.*?>", "", text))
    doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
    [nouns.add(lemma) for lemma in lemmas]
    return nouns


def getlemmas(text):
    lemmas = set()
    text = html.unescape(re.sub(r"<.*?>", "", text))
    doc = nlp(text)
    words = [token.lemma_.lower() for token in doc]
    [lemmas.add(lemma) for lemma in words]
    return lemmas


def getsummary(obj):
    if obj["model"][:6] == "claude":
        text = obj["message"]["content"][0]["text"]
    elif obj["model"][:3] == "gpt":
        text = obj["summary"]
    return text


def summarynouns(obj):
    summary = getsummary(obj)
    return getnouns(summary)


def summarylemmas(obj):
    summary = getsummary(obj)
    return getlemmas(summary)


def gethittexts(obj, cites):
    texts = []
    for i, o in enumerate(obj["search"]["data"]["hits"]["hits"]):
        if i in cites:
            if "title" in o and "description" in o:
                texts.append(f"{o['title']}.  {o['description']}.")
            elif "title" in o:
                texts.append(o["title"])
            elif "description" in o:
                texts.append(o["description"])
    return texts


def hitlemmas(obj, cites):
    texts = gethittexts(obj, cites)
    return getlemmas("\n\n".join(texts))


def hitnouns(obj, cites):
    texts = gethittexts(obj, cites)
    return getnouns("\n\n".join(texts))


def load(type="concept"):
    models = {}
    with jsonlines.open("interleaving-summaries.jsonl", "r") as jl:
        lst = [obj for obj in jl]
    for obj in lst:
        if obj["model"] not in models.keys():
            models[obj["model"]] = []
        cites = getcites(getsummary(obj))
        hn = hitnouns(obj, cites) if type == "concept" else hitlemmas(obj, cites)
        if len(hn) > 0:
            sn = summarynouns(obj) if type == "concept" else summarylemmas(obj)
            if len(sn) > 0:
                precision = 1 - (len(sn - hn) / len(sn))  # inverse of included nouns
                recall = 1 - (len(hn - sn) / len(hn))  # inverse of missing nouns
                models[obj["model"]].append((recall, precision))
                print(precision, recall)

    makerecallplot(models, type=type)
    makeprecisionplot(models, type=type)
    makef1plot(models, type=type)

    with open(f"{type}_f1.pickle", "wb") as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)


def report(type="concept"):
    fn = f"{type}_f1.pickle"
    with open(fn, "rb") as handle:
        models = pickle.load(handle)

    makerecallplot(models, type=type)
    makeprecisionplot(models, type=type)
    makef1plot(models, type=type)


def main():
    print("Hello from interleaving-rag!")
    # load(type="lemma")
    report(type="lemma")


if __name__ == "__main__":
    main()
