"""Public plot export contracts, including complex Scanpy plot objects."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import cellscope as cs
from cellscope.plotting._output import figures


@pytest.fixture
def rna():
    data = cs.datasets.toy_rna(n_cells=60, n_genes=20)
    cs.pp.qc(data)
    data.obsm['X_umap'] = np.random.default_rng(0).normal(size=(60, 2))
    data.uns['pca'] = {'variance_ratio': np.geomspace(.1, .001, 10)}
    return data


@pytest.mark.parametrize('name', ['embedding','dotplot','heatmap','qc','cell_composition','volcano','cell_ratio','dimplot','pca_variance_ratio'])
def test_native_object_and_optional_multi_format_output(name, rna, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def draw(**options):
        if name == 'dotplot':
            return cs.pl.dotplot(rna, ['G1','G2'], groupby='cell_type', **options)
        if name == 'heatmap':
            return cs.pl.heatmap(rna, ['G1','G2'], groupby='cell_type', **options)
        if name == 'qc':
            return cs.pl.qc(rna, keys=['total_counts'], groupby='sample_id', stripplot=False,
                            xlabel='Sample',ylabel='UMI',title='QC',ylim=(0,None), **options)
        if name == 'cell_composition':
            table = pd.DataFrame({'sample_id':['a','a','b','b'],'group':['T','B','T','B'],'fraction':[.7,.3,.6,.4]})
            return cs.pl.cell_composition(table, **options)
        if name == 'volcano':
            return cs.pl.volcano(pd.DataFrame({'log2FoldChange':[1,-2],'padj':[.01,.03]}), **options)
        if name == 'cell_ratio':
            return cs.pl.cell_ratio(rna, 'sample_id', 'cell_type', **options)
        if name == 'dimplot':
            return cs.pl.dimplot(rna, 'umap', color='cell_type', **options)
        if name == 'embedding':
            return cs.pl.embedding(rna,color=['cell_type','sample_id'], **options)
        return cs.pl.pca_variance_ratio(rna,**options)
    for parameter in ['filename','outdir','formats','dpi']:
        assert parameter in inspect.signature(getattr(cs.pl,name)).parameters
    result=draw()
    assert figures(result)
    assert not list(tmp_path.iterdir())
    for fig in figures(result):
        fig.savefig(tmp_path/'manual.png')
    (tmp_path/'manual.png').unlink()
    plt.close('all')
    # Ensure export never saves a different pyplot-current figure.
    result=draw(filename='result',outdir=tmp_path,formats=('pdf','png'),dpi=90)
    assert (tmp_path/'result.pdf').read_bytes().startswith(b'%PDF-')
    assert (tmp_path/'result.png').read_bytes().startswith(b'\x89PNG')
    assert figures(result)
    plt.close('all')


def test_batch_workflow_returns_plots_without_implicit_output(rna,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    result=cs.pl.plot_batch_effect(rna,batch_key='sample_id',cluster_key='review',n_pcs=5,mask_var=None)
    assert {'batch','cluster','pca_variance'} <= set(result)
    assert all(figures(plot) for plot in result.values())
    assert not list(tmp_path.iterdir())
    plt.close('all')


def test_export_uses_given_axes_not_current_figure(rna,tmp_path):
    fig,axis=plt.subplots()
    other=plt.figure()
    assert cs.pl.qc(rna,keys=['total_counts'],ax=axis,stripplot=False,filename='qc',outdir=tmp_path,formats='png') is axis
    assert axis.figure is fig and other is not fig
    assert (tmp_path/'qc.png').exists()
    plt.close('all')



def test_multi_metric_qc_and_show_still_return_objects(rna, tmp_path, monkeypatch):
    shown=[]
    monkeypatch.setattr(plt,'show',lambda:shown.append(True))
    result=cs.pl.qc(rna,groupby='sample_id',stripplot=False,show=True)
    assert shown and figures(result)
    assert not list(tmp_path.iterdir())
    result=cs.pl.dimplot(rna,'umap',color='sample_id',return_fig=True)
    assert hasattr(result,'savefig')
    plt.close('all')



def test_scanpy_global_autosave_cannot_bypass_filename(rna,tmp_path,monkeypatch):
    import scanpy as sc

    previous = sc.settings.autosave, sc.settings.figdir
    try:
        sc.settings.autosave = True
        sc.settings.figdir = tmp_path
        cs.pl.embedding(rna,color='sample_id')
        cs.pl.qc(rna,keys=['total_counts'],stripplot=False)
        cs.pl.dimplot(rna,'umap',color='sample_id')
        assert not list(tmp_path.iterdir())
    finally:
        sc.settings.autosave, sc.settings.figdir = previous
        plt.close('all')
