from bisect import bisect_left
import xml.etree.ElementTree as ET

tree = ET.parse('dc_map.xml')
root = tree.getroot()

associations = {}

for association in root:
    Lc = float(association.find('design/value').text)
    filename = str(association.find('extracted/value').text)
    associations[Lc] = filename

def take_closest(sorted_list, value):
    """
    Assumes ``sorted_list`` is sorted. Returns closest value to ``value``.

    If two numbers are equally close, return the smallest number.

    References
    ----------
    https://stackoverflow.com/a/12141511/11530613
    """
    pos = bisect_left(associations, Lc)
    if pos == 0:
        return associations[0]
    if pos == len(associations):
        return associations[-1]
    before = associations[pos - 1]
    after = associations[pos]
    if after - Lc < Lc - before:
       return after
    else:
       return before

associations[take_closest(list(associations.keys()), 1.35e-5)]