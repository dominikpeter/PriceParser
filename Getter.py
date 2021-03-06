import argparse
import io
import multiprocessing
import os
import zipfile
from functools import partial
import _main as pp

import requests
from lxml import html
from tqdm import tqdm

import _main as pp


def get_download_url(html_content, baseUrl=""):
    urls = []
    urls_title = {}
    for ref in html_content.xpath('//*[@class = "col_1"]'):
        for i in ref.xpath('//a'):
            try:
                if i.attrib['title'] == 'Download Katalog':
                    urls += [str(baseUrl) + str(i.attrib['href'])]
            except(KeyError):
                continue
    for title, url in zip(html_content.xpath('//*[@class="col_2"]'), urls):
        urls_title[title.text] = url
    return urls_title


def save_xml_from_url(path, url):
    r = requests.get(url[1], verify=False)
    _path = os.path.join(path, str(url[0]).replace('/', '').replace(r'\\', ''))
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(_path)


def extract_to_path(d, path, n_jobs_=0):
    if n_jobs_>0:
        func = partial(save_xml_from_url, path)
        pool = multiprocessing.Pool()
        pool.map(func, d.items())
    else:
        for i in tqdm(d.items()):
            save_xml_from_url(path, i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--n_jobs',default=0, dest="n_jobs",
        help="Number of Parallel Jobs", type=int)

    print("\n\n\n==========================================================\n\n",
        "Getting XML Files from {}".format(pp.Page),
        "\n\n""==========================================================\n\n\n")

    currentpath = pp.Path

    args = parser.parse_args()
    n_jobs = args.n_jobs

    r = requests.get(pp.Page, verify=False)
    html_content = html.fromstring(r.content)
    g = get_download_url(html_content, html_content.base)

    pp.create_folder(currentpath, 'XML')

    path = os.path.join(currentpath, 'XML')

    extract_to_path(g, path, n_jobs_=n_jobs)

    print("\n\n\nFinished...")
