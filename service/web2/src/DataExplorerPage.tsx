import React, { useState, useMemo } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Stack,
    CircularProgress,
    Card,
    CardContent,
    CardHeader,
    Divider,
    Grid,
    InputAdornment
} from '@mui/material';
import { Search } from 'lucide-react';
import { DataFlight, TrackPoint } from './types';
import { fetchDataFlights, fetchUnifiedTrack } from './api';

export const DataExplorerPage: React.FC = () => {
    return (
        <Box sx={{ p: 3, height: '100vh', overflow: 'auto', bgcolor: '#f5f5f5' }}>
            <Typography variant="h4" gutterBottom>
                Data Explorer
            </Typography>
            
            <Stack spacing={4}>
                <FlightsExplorer />
                <Divider />
                <TrackExplorer />
            </Stack>
        </Box>
    );
};

// Simple Bar Chart Component
const SimpleBarChart: React.FC<{ data: { label: string; value: number }[]; title: string; color?: string }> = ({ data, title, color = '#3b82f6' }) => {
    const maxValue = Math.max(...data.map(d => d.value), 1);
    
    return (
        <Card sx={{ height: '100%' }}>
            <CardHeader title={title} titleTypographyProps={{ variant: 'h6' }} />
            <CardContent>
                <div className="flex flex-col gap-2 h-64 overflow-y-auto pr-2">
                    {data.map((item, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm">
                            <div className="w-24 truncate text-right font-medium" title={item.label}>{item.label}</div>
                            <div className="flex-1 h-6 bg-gray-100 rounded overflow-hidden relative">
                                <div 
                                    className="h-full rounded transition-all duration-500"
                                    style={{ width: `${(item.value / maxValue) * 100}%`, backgroundColor: color }}
                                />
                                <span className="absolute inset-y-0 right-2 flex items-center text-xs text-gray-500">
                                    {item.value}
                                </span>
                            </div>
                        </div>
                    ))}
                    {data.length === 0 && <p className="text-center text-gray-400 mt-10">No data</p>}
                </div>
            </CardContent>
        </Card>
    );
};

const FlightsExplorer: React.FC = () => {
    // Default to last 24 hours
    const [startDate, setStartDate] = useState(new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().slice(0, 16));
    const [endDate, setEndDate] = useState(new Date().toISOString().slice(0, 16));
    const [flights, setFlights] = useState<DataFlight[]>([]);
    const [loading, setLoading] = useState(false);
    const [sortField, setSortField] = useState<keyof DataFlight>('start_time');
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

    // Filters
    const [filters, setFilters] = useState({
        flight_id: '',
        callsign: '',
        source: ''
    });

    const handleSearch = async () => {
        setLoading(true);
        try {
            const startTs = Math.floor(new Date(startDate).getTime() / 1000);
            const endTs = Math.floor(new Date(endDate).getTime() / 1000);
            const data = await fetchDataFlights(startTs, endTs);
            setFlights(data);
        } catch (error) {
            console.error(error);
            alert('Failed to fetch flights');
        } finally {
            setLoading(false);
        }
    };

    const handleSort = (field: keyof DataFlight) => {
        if (sortField === field) {
            setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortDirection('desc'); // Default to desc for new fields usually
        }
    };

    const filteredFlights = useMemo(() => {
        return flights.filter(f => {
            const matchId = f.flight_id.toLowerCase().includes(filters.flight_id.toLowerCase());
            const matchCallsign = (f.callsign || '').toLowerCase().includes(filters.callsign.toLowerCase());
            const matchSource = f.source.toLowerCase().includes(filters.source.toLowerCase());
            return matchId && matchCallsign && matchSource;
        });
    }, [flights, filters]);

    const sortedFlights = useMemo(() => {
        return [...filteredFlights].sort((a, b) => {
            const aVal = a[sortField];
            const bVal = b[sortField];
            
            if (aVal === undefined || bVal === undefined) return 0;
            
            if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
            if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }, [filteredFlights, sortField, sortDirection]);

    // Aggregations
    const aggregations = useMemo(() => {
        const bySource: Record<string, number> = {};
        const byCallsign: Record<string, number> = {};

        filteredFlights.forEach(f => {
            // Source
            bySource[f.source] = (bySource[f.source] || 0) + 1;
            
            // Callsign
            const cs = f.callsign || 'Unknown';
            byCallsign[cs] = (byCallsign[cs] || 0) + 1;
        });

        const sourceData = Object.entries(bySource)
            .map(([label, value]) => ({ label, value }))
            .sort((a, b) => b.value - a.value);

        const callsignData = Object.entries(byCallsign)
            .map(([label, value]) => ({ label, value }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 10); // Top 10

        return { sourceData, callsignData };
    }, [filteredFlights]);

    return (
        <Stack spacing={3}>
            {/* Aggregation Graphs */}
            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 6 }}>
                    <SimpleBarChart data={aggregations.sourceData} title="Flights by Source" color="#10b981" />
                </Grid>
                <Grid size={{ xs: 12, md: 6 }}>
                    <SimpleBarChart data={aggregations.callsignData} title="Top 10 Callsigns" color="#6366f1" />
                </Grid>
            </Grid>

            <Card>
                <CardHeader title="Flights List" subheader="Search and filter flight metadata" />
                <CardContent>
                    <Stack direction="row" spacing={2} sx={{ mb: 2 }} alignItems="center">
                        <TextField
                            label="Start Date"
                            type="datetime-local"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            InputLabelProps={{ shrink: true }}
                            size="small"
                        />
                        <TextField
                            label="End Date"
                            type="datetime-local"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            InputLabelProps={{ shrink: true }}
                            size="small"
                        />
                        <Button variant="contained" onClick={handleSearch} disabled={loading}>
                            {loading ? <CircularProgress size={24} color="inherit" /> : 'Search Flights'}
                        </Button>
                    </Stack>

                    <Typography variant="body2" sx={{ mb: 1 }}>
                        Found {filteredFlights.length} flights
                    </Typography>

                    <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
                        <Table stickyHeader size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer mb-1" onClick={() => handleSort('flight_id')}>Flight ID</div>
                                        <TextField 
                                            placeholder="Filter..." 
                                            variant="standard" 
                                            size="small"
                                            value={filters.flight_id}
                                            onChange={(e) => setFilters(prev => ({ ...prev, flight_id: e.target.value }))}
                                            InputProps={{
                                                startAdornment: <InputAdornment position="start"><Search className="w-3 h-3" /></InputAdornment>,
                                                disableUnderline: true,
                                                className: "bg-gray-50 px-2 rounded"
                                            }}
                                        />
                                    </TableCell>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer mb-1" onClick={() => handleSort('callsign')}>Callsign</div>
                                        <TextField 
                                            placeholder="Filter..." 
                                            variant="standard" 
                                            size="small"
                                            value={filters.callsign}
                                            onChange={(e) => setFilters(prev => ({ ...prev, callsign: e.target.value }))}
                                            InputProps={{
                                                startAdornment: <InputAdornment position="start"><Search className="w-3 h-3" /></InputAdornment>,
                                                disableUnderline: true,
                                                className: "bg-gray-50 px-2 rounded"
                                            }}
                                        />
                                    </TableCell>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer" onClick={() => handleSort('start_time')}>Start Time</div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer" onClick={() => handleSort('end_time')}>End Time</div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer" onClick={() => handleSort('point_count')}>Points</div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="font-bold cursor-pointer mb-1" onClick={() => handleSort('source')}>Source</div>
                                        <TextField 
                                            placeholder="Filter..." 
                                            variant="standard" 
                                            size="small"
                                            value={filters.source}
                                            onChange={(e) => setFilters(prev => ({ ...prev, source: e.target.value }))}
                                            InputProps={{
                                                startAdornment: <InputAdornment position="start"><Search className="w-3 h-3" /></InputAdornment>,
                                                disableUnderline: true,
                                                className: "bg-gray-50 px-2 rounded"
                                            }}
                                        />
                                    </TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {sortedFlights.map((f) => (
                                    <TableRow key={f.flight_id} hover>
                                        <TableCell sx={{ fontFamily: 'monospace' }}>{f.flight_id}</TableCell>
                                        <TableCell>{f.callsign || '-'}</TableCell>
                                        <TableCell>{new Date(f.start_time * 1000).toLocaleString()}</TableCell>
                                        <TableCell>{new Date(f.end_time * 1000).toLocaleString()}</TableCell>
                                        <TableCell>{f.point_count}</TableCell>
                                        <TableCell>{f.source}</TableCell>
                                    </TableRow>
                                ))}
                                {flights.length === 0 && !loading && (
                                    <TableRow>
                                        <TableCell colSpan={6} align="center">No flights found</TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </CardContent>
            </Card>
        </Stack>
    );
};

const TrackExplorer: React.FC = () => {
    const [flightId, setFlightId] = useState('');
    const [points, setPoints] = useState<TrackPoint[]>([]);
    const [loading, setLoading] = useState(false);
    const [fetchedId, setFetchedId] = useState('');

    const handleSearch = async () => {
        if (!flightId) return;
        setLoading(true);
        try {
            const track = await fetchUnifiedTrack(flightId);
            setPoints(track.points || []);
            setFetchedId(track.flight_id);
        } catch (error) {
            console.error(error);
            alert('Failed to fetch track');
            setPoints([]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card>
            <CardHeader title="Track Explorer" subheader="View raw track points for a specific flight" />
            <CardContent>
                <Stack direction="row" spacing={2} sx={{ mb: 2 }} alignItems="center">
                    <TextField
                        label="Flight ID"
                        value={flightId}
                        onChange={(e) => setFlightId(e.target.value)}
                        placeholder="e.g. 3d3af8dd"
                        size="small"
                    />
                    <Button variant="contained" onClick={handleSearch} disabled={loading || !flightId}>
                        {loading ? <CircularProgress size={24} /> : 'Load Track'}
                    </Button>
                </Stack>

                {fetchedId && (
                    <Typography variant="body2" sx={{ mb: 1 }}>
                        Showing {points.length} points for Flight {fetchedId}
                    </Typography>
                )}

                <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                    <Table stickyHeader size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Timestamp</TableCell>
                                <TableCell>Time</TableCell>
                                <TableCell>Lat</TableCell>
                                <TableCell>Lon</TableCell>
                                <TableCell>Alt (ft)</TableCell>
                                <TableCell>Speed (kts)</TableCell>
                                <TableCell>Heading</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {points.map((p, i) => (
                                <TableRow key={i} hover>
                                    <TableCell>{p.timestamp}</TableCell>
                                    <TableCell>{new Date(p.timestamp * 1000).toLocaleTimeString()}</TableCell>
                                    <TableCell>{p.lat.toFixed(5)}</TableCell>
                                    <TableCell>{p.lon.toFixed(5)}</TableCell>
                                    <TableCell>{p.alt}</TableCell>
                                    <TableCell>{p.gspeed ?? '-'}</TableCell>
                                    <TableCell>{p.track ?? '-'}</TableCell>
                                </TableRow>
                            ))}
                            {points.length === 0 && !loading && (
                                <TableRow>
                                    <TableCell colSpan={7} align="center">No points to display</TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </CardContent>
        </Card>
    );
};
