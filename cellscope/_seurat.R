function(path, assay, requested, reductions) {
    if (utils::packageVersion("SeuratObject") < "5.0.0" ||
        utils::packageVersion("Seurat") < "5.0.0")
        stop("Seurat RDS import requires Seurat and SeuratObject >= 5.0.0")
    object <- readRDS(path)
    if (!inherits(object, "Seurat")) stop("RDS must contain a Seurat object")
    if (!assay %in% SeuratObject::Assays(object))
        stop(paste0("Assay not found: ", assay))
    a <- object[[assay]]
    available <- SeuratObject::Layers(a, search=NA)
    selected <- unique(unlist(lapply(requested, function(key) {
        if (key %in% available) key else available[startsWith(available, paste0(key, "."))]
    })))
    matrices <- setNames(lapply(selected, function(key) {
        x <- SeuratObject::LayerData(a, layer=key, fast=FALSE)
        # Materialize disk-backed matrices as sparse, never as a dense matrix.
        if (!is.matrix(x) && !inherits(x, "sparseMatrix")) x <- as(x, "dgCMatrix")
        list(matrix=x, features=rownames(x), cells=colnames(x))
    }), selected)
    pack_metadata <- function(x) {
        factors <- lapply(x[vapply(x, is.factor, logical(1))], function(v)
            list(levels=levels(v), ordered=is.ordered(v)))
        if (any(vapply(x, is.list, logical(1))))
            stop("List-valued metadata columns cannot be converted to AnnData")
        list(frame=x, factors=factors)
    }
    obs <- object[[]]
    if ("seurat_ident" %in% colnames(obs))
        stop("Metadata column 'seurat_ident' is reserved for active identities")
    obs$seurat_ident <- SeuratObject::Idents(object)[rownames(obs)]
    embeddings <- list()
    if (reductions) for (key in SeuratObject::Reductions(object)) {
        reduction <- object[[key]]
        if (SeuratObject::DefaultAssay(reduction) != assay) next
        x <- SeuratObject::Embeddings(reduction)
        embeddings[[key]] <- list(matrix=x, cells=rownames(x))
    }
    list(matrices=matrices, obs=pack_metadata(obs), var=pack_metadata(a[[]]),
         variable_features=as.character(SeuratObject::VariableFeatures(a)), reductions=embeddings,
         available_layers=available, object_version=as.character(object@version),
         seurat_version=as.character(utils::packageVersion("Seurat")),
         seuratobject_version=as.character(utils::packageVersion("SeuratObject")))
}
