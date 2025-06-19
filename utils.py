# utils.py

from bs4 import BeautifulSoup

def insert_tables_into_html(html_content: str, tables: list) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    
    for i, table_data in enumerate(tables):
        table_tag = soup.find("table", {"id": f"autofill-table-{i+1}"})
        if not table_tag:
            continue

        # Clear old rows
        for row in table_tag.find_all("tr"):
            row.decompose()

        # Add new rows
        for row in table_data:
            tr = soup.new_tag("tr")
            for cell in row:
                td = soup.new_tag("td")
                td.string = str(cell)
                tr.append(td)
            table_tag.append(tr)

    return str(soup)
