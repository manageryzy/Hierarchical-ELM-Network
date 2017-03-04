function b = bindata(x)
b = 0;
if x == edges(nEdges)
    b = nEdges;
else
    for idx = 1:(nEdges-1)
        if x >= edges(idx) && x < edges(idx + 1)
            b = idx;
            break;
        end
    end
end
