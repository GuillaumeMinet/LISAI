import os
from pathlib import Path
import warnings

def create_save_folder(path,limit=20,overwrite=False,parent_exists_check=False):

    """
    Utils function creating folders with path_i where i is automatically 
    updated to +1 if folder already exists and overwrite=False, until we hit limit.

    Parameters
    ----------
    path: PATH object
        path of folder that we want to create
    limit: int, default=20
        maximum idx value of "<path>_<idx>" before raising exception.
    overwrite: bool, default=False
        if True and path exists, we don't recreate one. That means files
        in there might be overwritten.
    parent_exists_check: bool, default = False,
        if True, only creates if parent folder already exists.

    Output
    -------
    new_path : PATH object.
        saving folder path

    """

    path = Path(path)

    if overwrite and os.path.exists(path):
        warnings.warn(f"Existing files in save folder {path} might get overwritten")
        return path
    
    if parent_exists_check and not os.path.exists(path.parent):
        return None
    
    i = 0
    while i < limit:
        if i==0:
            new_path = path
        else:
            new_path = Path(str(path) + f"_{i:02d}")
            
        if os.path.exists(new_path):
            i += 1
        else:
            os.makedirs(new_path)
            return new_path
        
    # raise exception if i goes beyond limit
    raise Exception ("Name for saving folder, change experiment name!")


def replace_dict(dict1,dict2,keyword='same'):
    """
    Utils function to replace items in dict1 with items of dict2
    for items that have the same key, unless item == keyword, where
    in that case nothing is done.

    Use case is: you have a config file with all the parameters, and
    you just want to update some of the parameters with another config
    file.

    NOTE: could be updated where keyword is a list if needed to exclude 
    more than one keyword.

    NOTE: works for nested dictionaries, but not for nested tuples or lists,
    and only if different keys between dictionary are at the same nested level.

    inputs:
        - dict1 : dictionary 1.
        - dict2 : dictionary 2.
        - keyword (opt - default = 'same')
    output:
        - dict1 : dict1 with updated items from dict2.
    """


    for key in dict1:
        if key in dict2:
            if type(dict1[key])==dict:
                replace_dict(dict1[key],dict2[key])
            elif dict2[key] != keyword:
                dict1[key] = dict2[key]

    return dict1


def getDivisors(n) : 
    divisors = []
    i = 1
    while i <= n : 
        if (n % i==0) : 
            divisors.append(i), 
        i = i + 1
    return divisors


def find_closest_divisor(a,b,smaller=False):
    """
    Rounds "b" to the closest divisor making it a divisor of "a".
    If smaller = True, it forces a value < b.
    """
    if a % b == 0:
        return b
    all  = getDivisors(a)
    for idx,val in enumerate(all):
        if b < val:
            if idx == 0:
                return b
            if smaller or (val-b)>(b-all[idx-1]):
                return all[idx-1]
            return val


def nested_get(d:dict, keys:list):
    """
    To get keys in a nested dictionnary.
    """
    if not isinstance(keys,list):
        keys = [keys]
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None  # If the path is invalid
    return d



def nested_replace(d, keys, item):
    """
    To replace item in a nested dictionnary, accessed
    by a list of keys.
    """

    if not isinstance(keys, list):  # Convert string key to list
        keys = [keys]
    
    current = d
    for key in keys[:-1]:  # Traverse up to the second-last key
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False  # Key path not found
    
    last_key = keys[-1]
    if isinstance(current, dict) and last_key in current:
        current[last_key] = item
        return True  # Successfully replaced
    
    return False  # Key path not found


if __name__ == "__main__":
    a = 1410
    b = 200
    c = find_closest_divisor(a,b,True)
    print(c)