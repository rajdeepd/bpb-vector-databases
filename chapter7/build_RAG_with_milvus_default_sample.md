<a href="https://colab.research.google.com/github/rajdeepd/bpb-vector-databases/blob/main/chapter7/build_RAG_with_milvus_default_sample.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/build_RAG_with_milvus.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>   <a href="https://colab.research.google.com/github/rajdeepd/bpb-vector-databases/blob/main/chapter7/build_RAG_with_milvus_default_sample.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on%20GitHub-555555?style=flat&logo=github&logoColor=white" alt="GitHub Repository"/>

# Build RAG with Milvus

<img src="https://raw.githubusercontent.com/milvus-io/bootcamp/master/bootcamp/tutorials/quickstart/apps/rag_search_with_milvus/pics/rag_demo.png"/>

In this section, we will show you how to build a RAG(Retrieval-Augmented Generation) pipeline with Milvus.

The RAG system combines a retrieval system with a generative model to generate new text based on a given prompt. The system first retrieves relevant documents from a corpus using Milvus, and then uses a generative model to generate new text based on the retrieved documents.


## Preparation
### Dependencies and Environment


```python
! pip install --upgrade -q pymilvus openai requests tqdm
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.7/57.7 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m212.8/212.8 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m389.6/389.6 kB[0m [31m22.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.5/78.5 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.9/5.9 MB[0m [31m29.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.4/49.4 MB[0m [31m14.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.6/53.6 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25h

> If you are using Google Colab, to enable dependencies just installed, you may need to **restart the runtime** (click on the "Runtime" menu at the top of the screen, and select "Restart session" from the dropdown menu).

We will use OpenAI as the LLM in this example. You should prepare the [api key](https://platform.openai.com/docs/quickstart) `OPENAI_API_KEY` as an environment variable.


```python
import os
from google.colab import userdata
openai_api_key = userdata.get('OPENAI_API_KEY')
print(openai_api_key[:20])
#os.environ["OPENAI_API_KEY"] = "sk-***********"
os.environ["OPENAI_API_KEY"] = openai_api_key
```

    sk-proj-wuN1uyV_WAUJ


### Prepare the data

We use the FAQ pages from the [Milvus Documentation 2.4.x](https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip) as the private knowledge in our RAG, which is a good data source for a simple RAG pipeline.

Download the zip file and extract documents to the folder `milvus_docs`.


```python
! wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
! unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
```

    --2024-11-30 05:41:26--  https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
    Resolving github.com (github.com)... 140.82.113.3
    Connecting to github.com (github.com)|140.82.113.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/267273319/c52902a0-e13c-4ca7-92e0-086751098a05?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241130T054127Z&X-Amz-Expires=300&X-Amz-Signature=57e559fa7fd828a6afcfb3472461fa56bd89f9d25c3dcaaaff78d467fd651fb4&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dmilvus_docs_2.4.x_en.zip&response-content-type=application%2Foctet-stream [following]
    --2024-11-30 05:41:27--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/267273319/c52902a0-e13c-4ca7-92e0-086751098a05?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241130T054127Z&X-Amz-Expires=300&X-Amz-Signature=57e559fa7fd828a6afcfb3472461fa56bd89f9d25c3dcaaaff78d467fd651fb4&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dmilvus_docs_2.4.x_en.zip&response-content-type=application%2Foctet-stream
    Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 613094 (599K) [application/octet-stream]
    Saving to: ‘milvus_docs_2.4.x_en.zip’
    
    milvus_docs_2.4.x_e 100%[===================>] 598.72K  --.-KB/s    in 0.04s   
    
    2024-11-30 05:41:27 (14.1 MB/s) - ‘milvus_docs_2.4.x_en.zip’ saved [613094/613094]
    



```python
ls ./milvus_docs/en/adminGuide
```

    allocate.md               dynamic_config.md            scaleout.md
    authenticate.md           limit_collection_counts.md   security.md
    [0m[01;34mbackup[0m/                   m2m_legacy.md                tls.md
    chunk_cache.md            manage_dependencies.md       upgrade_milvus_cluster-docker.md
    [01;34mclouds[0m/                   message_storage_operator.md  upgrade_milvus_cluster-helm.md
    configure_access_logs.md  meta_storage_operator.md     upgrade_milvus_cluster-operator.md
    configure-docker.md       [01;34mmonitor[0m/                     upgrade_milvus_embedded.md
    configure-helm.md         object_storage_operator.md   upgrade_milvus_standalone-docker.md
    configure_operator.md     operator.md                  upgrade_milvus_standalone-helm.md
    deploy_etcd.md            rbac.md                      upgrade_milvus_standalone-operator.md
    deploy_pulsar.md          resource_group.md
    deploy_s3.md              scale-dependencies.md


We load all markdown files from the folder `milvus_docs/en/faq`. For each document, we just simply use "# " to separate the content in the file, which can roughly separate the content of each main part of the markdown file.


```python
from glob import glob

text_lines = []
#"milvus_docs/en/faq/*.md"
admin_guides_path = './milvus_docs/en/adminGuide/*.md'
for file_path in glob(admin_guides_path, recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")
```

### Prepare the Embedding Model

We initialize the OpenAI client to prepare the embedding model.


```python
from openai import OpenAI

openai_client = OpenAI()
```

Define a function to generate text embeddings using OpenAI client. We use the [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) model as an example.


```python
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
```

Generate a test embedding and print its dimension and first few elements.


```python
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
```

    1536
    [0.009889289736747742, -0.005578675772994757, 0.00683477520942688, -0.03805781528353691, -0.01824733428657055, -0.04121600463986397, -0.007636285852640867, 0.03225184231996536, 0.018949154764413834, 9.352207416668534e-05]


## Load data into Milvus

### Create the Collection


```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "admin_guide_rag_collection"
```

    DEBUG:pymilvus.milvus_client.milvus_client:Created new connection using: e8b68136e4f746919a97175ed2adc41e


> As for the argument of `MilvusClient`:
> - Setting the `uri` as a local file, e.g.`./milvus.db`, is the most convenient method, as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file.
> - If you have large scale of data, you can set up a more performant Milvus server on [docker or kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server uri, e.g.`http://localhost:19530`, as your `uri`.
> - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.

Check if the collection already exists and drop it if it does.


```python
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
```

Create a new collection with specified parameters.

If we don't specify any field information, Milvus will automatically create a default `id` field for primary key, and a `vector` field to store the vector data. A reserved JSON field is used to store non-schema-defined fields and their values.


```python
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)
```

    DEBUG:pymilvus.milvus_client.milvus_client:Successfully created collection: admin_guide_rag_collection
    DEBUG:pymilvus.milvus_client.milvus_client:Successfully created an index on collection: admin_guide_rag_collection


### Insert data
Iterate through the text lines, create embeddings, and then insert the data into Milvus.

Here is a new field `text`, which is a non-defined field in the collection schema. It will be automatically added to the reserved JSON dynamic field, which can be treated as a normal field at a high level.


```python
from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
```

    Creating embeddings: 100%|██████████| 730/730 [02:31<00:00,  4.83it/s]





    {'insert_count': 730, 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729], 'cost': 0}



## Build RAG

### Retrieve data for a query

Let's specify a frequent question about Milvus.


```python
question = "How does Milvus handle access control?"
```

Search for the question in the collection and retrieve the semantic top-3 matches.


```python
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)
```

Let's take a look at the search results of the query



```python
import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
```

    [
        [
            "Security\n\nMilvus attempts to ensure database security both on the access level and on the database level.\n\n- [Enable Authentication](authenticate.md): Milvus supports authenticated access by username and password.\n\n- [Encryption in Transit](tls.md): TLS (Transport Layer Security) is an encryption protocol to ensure communication security. \n",
            0.684380292892456
        ],
        [
            "Enable RBAC\n\nBy enabling RBAC, you can control access to specific Milvus resources (Eg. a collection or a partition) or permissions based on user role and privileges. Currently, this feature is only available in Python and Java.\n\nThis topic describes how to enable RBAC and manage [users and roles](users_and_roles.md).\n\n<div class=\"alert note\">\n\nThe code snippets on this page use new <a href=\"https://milvus.io/api-reference/pymilvus/v2.4.x/About.md\">MilvusClient</a> (Python) to interact with Milvus. New MilvusClient SDKs for other languages will be released in future updates.\n\n</div>\n\n#",
            0.6203930974006653
        ],
        [
            "Configure Milvus on the Fly\n\nMilvus allows you to change some of its configurations on the fly.\n\n#",
            0.614767849445343
        ]
    ]


### Use LLM to get a RAG response

Convert the retrieved documents into a string format.


```python
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
```

Define system and user prompts for the Lanage Model. This prompt is assembled with the retrieved documents from Milvus.


```python
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
```

Use OpenAI ChatGPT to generate a response based on the prompts.


```python
response = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
```

    Milvus handles access control through various mechanisms including enabling authentication with username and password, encryption in transit using TLS (Transport Layer Security), and enabling Role-Based Access Control (RBAC) to control access to specific resources based on user roles and privileges.





