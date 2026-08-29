MATCH (n:Person) WHERE n.name = $name RETURN n ORDER BY n.name;
MATCH (n:`Weird Label`) WHERE n.note = 'CALL apoc.bad();' RETURN n;
CALL apoc.create.uuid();
WIBBLE (n) RETURN n;
MATCH (n) RETURN vendorMagic(n);
MATCH (n) WHERE n.note = 'unterminated
