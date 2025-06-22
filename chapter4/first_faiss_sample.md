<a href="https://colab.research.google.com/github/rajdeepd/bpb-vector-databases/blob/main/chapter4/first_faiss_sample.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Reference: https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772



```python
!pip install -q  faiss-cpu
!pip install -q sentence-transformers
```

```python
import pandas as pd
data = [['Where are your headquarters located?', 'location'],
['Throw my cellphone in the water', 'random'],
['Network Access Control?', 'networking'],
['Address', 'location']]
df = pd.DataFrame(data, columns = ['text', 'category'])
```


```python
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Where are your headquarters located?</td>
      <td>location</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Throw my cellphone in the water</td>
      <td>random</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Network Access Control?</td>
      <td>networking</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Address</td>
      <td>location</td>
    </tr>
  </tbody>
</table>







```python
from sentence_transformers import SentenceTransformer
text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)
```


```python
import faiss

vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)
```


```python
import numpy as np

search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)
```


```python
k = index.ntotal
distances, ann = index.search(_vector, k=k)
```


```python
results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
```


```python
results
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distances</th>
      <th>ann</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.584873</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.175950</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.644265</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.919767</td>
      <td>1</td>
    </tr>
  </tbody>
</table>





```python

# join by: df1.ann == data.index
merge = pd.merge(results, df, left_on='ann',
right_index=True)

```


```python
merge
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distances</th>
      <th>ann</th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.584873</td>
      <td>0</td>
      <td>Where are your headquarters located?</td>
      <td>location</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.175950</td>
      <td>3</td>
      <td>Address</td>
      <td>location</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.644265</td>
      <td>2</td>
      <td>Network Access Control?</td>
      <td>networking</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.919767</td>
      <td>1</td>
      <td>Throw my cellphone in the water</td>
      <td>random</td>
    </tr>
  </tbody>
</table>









```python
labels = df['category']
category = labels[ann[0][0]]
```


```python
category
```

