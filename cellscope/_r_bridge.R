# Internal transport for isolated R execution. Arguments are data, never R code.
args <- commandArgs(trailingOnly=TRUE)
options(digits=17)
if (!requireNamespace("jsonlite", quietly=TRUE))
    stop("CELLSCOPE_MISSING_PACKAGES: jsonlite. Run install.packages('jsonlite') in R.")
request <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
missing <- Filter(function(p) !requireNamespace(p, quietly=TRUE), unlist(request$packages))
if (length(missing)) stop(paste0("CELLSCOPE_MISSING_PACKAGES: ", paste(missing, collapse=", "),
    ". Install CRAN packages with install.packages; Bioconductor packages with BiocManager::install. ",
    "See docs/best_practices.md."))
decode <- function(x) {
    switch(x$kind,
        null=NULL,
        scalar=x$value,
        vector={
            values <- lapply(x$values, function(v) if(is.null(v)) NA else v)
            result <- unlist(values, use.names=FALSE)
            switch(x$type, numeric=as.numeric(result), logical=as.logical(result),
                   character=as.character(result))
        },
        matrix={
            dims <- as.integer(unlist(x$shape))
            values <- scan(x$path, sep=",", quiet=TRUE, what=numeric())
            matrix(values, nrow=dims[[1]], ncol=dims[[2]], byrow=TRUE)
        },
        sparse=as(Matrix::readMM(x$path), "CsparseMatrix"),
        dataframe={
            columns <- lapply(x$columns, decode)
            result <- as.data.frame(columns, check.names=FALSE, stringsAsFactors=FALSE)
            if (!length(columns)) result <- data.frame(row.names=unlist(x$index))
            else rownames(result) <- unlist(x$index)
            result
        },
        list=lapply(x$items, decode),
        rds=readRDS(x$path),
        stop("Unsupported input kind")
    )
}
serial <- 0L
encode <- function(x) {
    serial <<- serial + 1L
    stem <- file.path(request$directory, paste0("output-", serial))
    if(is.null(x)) return(list(kind="null"))
    if(inherits(x, "sparseMatrix")) {
        path <- paste0(stem, ".mtx"); Matrix::writeMM(x, path)
        return(list(kind="sparse", path=path))
    }
    if(is.data.frame(x)) return(list(kind="dataframe", index=as.list(rownames(x)),
                                   columns=lapply(x, encode)))
    if(is.matrix(x) && is.numeric(x)) {
        path <- paste0(stem, ".csv")
        utils::write.table(x, path, sep=",", row.names=FALSE, col.names=FALSE, na="NaN")
        return(list(kind="matrix", path=path, shape=as.list(dim(x))))
    }
    if(isS4(x) || is.environment(x) || is.function(x)) {
        path <- paste0(stem, ".rds"); saveRDS(x, path)
        return(list(kind="rds", path=path, classes=as.list(class(x))))
    }
    if(is.list(x)) return(list(kind="list", items=lapply(x, encode)))
    if(is.factor(x)) x <- as.character(x)
    if(is.atomic(x) && is.null(dim(x))) return(list(kind="vector", type=typeof(x),
        values=unname(as.list(x)), names=if(is.null(names(x))) NULL else as.list(names(x))))
    path <- paste0(stem, ".rds"); saveRDS(x, path)
    list(kind="rds", path=path, classes=as.list(class(x)))
}
fun <- eval(parse(text=request[["function"]]), envir=new.env(parent=globalenv()))
result <- do.call(fun, lapply(request$arguments, decode))
jsonlite::write_json(encode(result), args[[2]], auto_unbox=TRUE, null="null", na="null", digits=NA)
