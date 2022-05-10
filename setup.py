from setuptools import setup, find_packages
setup(name='cyto_align',
      version='1.0.0',
      description='A Package for Flow Cytometry Alignment',
      author='Muhammad Saeed',
      author_email='mohammud.saeed.batekh@gmail.com',
      url='',
      py_modules=['segmentation', 'alignment', 'helpers', 'visualization', 'pipeline', 'grouping'],
      package_dir ={'':'src/cyto'},
      packages = find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          'statsmodels',
          'scikit-image',
          'scikit-learn',
          'seaborn',
          'networkx',
          'jenkspy',
          'sklearn-contrib-py-earth',
          'simpleai'
        ]
     )

