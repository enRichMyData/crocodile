# GraphQL Queries for Crocodile Repository

## Get All Datasets (Paginated)

```graphql
{
  getDatasets(pageSize: 3) {
    datasets {
      id
      datasetName
      status
    }
    prevCursor
    nextCursor 
    hasNextPage
  }
}
```

```graphql
{
  getDatasets(cursor: "67a6971925a5cedd89917c43", pageSize: 3, direction: "next") {
    datasets {
      id
      datasetName
      status
    }
    prevCursor
    nextCursor 
    hasNextPage
  }
}
```

---

## Get Specific Dataset Info

```graphql
{
  getDatasetInfo(datasetName: "Round4_2020") {
    id
    datasetName
    status
    totalRows
    completionPercentage
  }
}
```

---

## Get Tables in a Dataset (Paginated)

```graphql
{
  getTablesInDataset(datasetName: "Round4_2020", pageSize: 2) {
    tables {
      id
      tableName
    }
    prevCursor
    nextCursor
    hasNextPage
  }
}
```

---

## Get Specific Table Info

```graphql
{
  getTableInfo(tableName: "HATOSLVE") {
    id
    tableName
    datasetName
    totalRows
    status
  }
}
```

---

## Get Table Data (Paginated)

```graphql
{
  getTableData(datasetName: "Round4_2020", tableName: "HATOSLVE", pageSize: 3) {
    rows {
      id
      rowId
      rowData
    }
    prevCursor
    nextCursor
    hasNextPage
  }
}
```

---

### 📌 Notes:
- Use **`cursor`** for pagination to fetch more datasets or tables.
- **`hasNextPage`** helps determine if more data is available.
- Replace **`datasetName`** and **`tableName`** as needed.

Happy querying! 🚀
