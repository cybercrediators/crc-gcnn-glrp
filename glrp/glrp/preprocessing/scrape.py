import urllib.request
import re
import tarfile
import gzip
import shutil
import glob
import os
from pathlib import Path
from tqdm import tqdm


class GeoScraper(object):
    def __init__(self, config):
        self.base_url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/'
        self.dataset_url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='
        self.config = config
        self.raw_fname_list = None

    def download(self, fpath, url):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, fpath, reporthook=t.update_to)

    def build_url(self):
        pass

    def download_code(self, code):
        """Download a given geo accession code into a subfolder of a given name"""
        target_fname = code + "_RAW.tar"
        code_no = re.sub(r'...$', 'nnn', code)
        target_url = self.base_url + code_no + "/" + code + "/suppl/" + target_fname
        dest_folder = self.config["base_data_folder"] + code + "/"
        dest_file = dest_folder + target_fname
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        if self.raw_fname_list is not None:
            self.raw_fname_list.append(dest_file)
        if (os.path.isfile(dest_file)):
            print("File " + dest_file + " was already downloaded!")
            return
        self.download(dest_file, target_url)

    def download_file(self, fname):
        """
        Download geo datasets from a given txt file.
        """
        for code in self.read_codes(fname):
            self.download_code(code)

    def unpack_tar(self, fname):
        """
        Unpack a given .tar archive.
        """
        tar = tarfile.open(fname)
        dest = Path(fname)
        #tar.extractall(self.config["base_data_folder"] )
        tar.extractall(dest.parents[0])
        tar.close()

    def unpack_cel(self, src, dest):
        """
        Unpack a given .cel.gz file
        """
        with gzip.open(src, 'rb') as s_file, open(dest, 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, 65536)

    def unpack_files(self):
        """
        Unpack all downloaded files in the current data folder
        """
        # unpack all tar raw files
        print("Start unpacking RAW files...")
        for f in self.raw_fname_list:
            self.unpack_tar(f)
        # TODO: change folder to the corresponding subfolder
        cel_files = glob.glob(self.config["base_data_folder"] + "**/*.gz", recursive=True)
        print("Start unpacking CEL files...")
        for f in cel_files:
            t = Path(f)
            dest = t.parent / t.stem
            self.unpack_cel(t, dest)

    def read_codes(self, fname):
        """read gse codes from a filename and return them"""
        with open(fname, 'r') as f:
            codes = f.read().splitlines()
        return codes

    def clean_raw_files(self):
        """remove downloaded raw tar files from the base data folder"""
        #endings = ["*.gz", "*.tar", "*.CHP"]
        endings = ["*.gz", "*.CHP"]
        for ext in endings:
            # TODO: take subfolders into account
            files = glob.glob(self.config["base_data_folder"] +"**/" + ext, recursive=True)
            for f in files:
                os.remove(f)

    def full_prepare_data(self, codes_fname):
        """Download and prepare RAW data for further preprocessing."""
        download_path = self.config["base_data_folder"]
        self.raw_fname_list = []
        # read codes, download codes
        self.download_file(codes_fname)
        #print(self.raw_fname_list)
        # unpack raw files
        # unpack cel files in folder
        self.unpack_files()
        # remove compressed files
        self.raw_fname_list.clear()
        self.clean_raw_files()

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# conf = {"base_data_folder": "./test_folder/"}
# scraper = GeoScraper(conf)
# scraper.download_code("GSE8671")
# scraper.full_prepare_data("./codes.txt")
